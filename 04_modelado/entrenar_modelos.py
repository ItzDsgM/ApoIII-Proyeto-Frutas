"""Pipeline reproducible de entrenamiento, evaluación y exportación del modelo final.

Ejecuta de extremo a extremo las Fases 4, 5 y preparación de Fase 6 de CRISP-DM:
1. Carga las particiones sin fuga ``data/annotations/fruitnet_splits.csv``.
2. Entrena y ajusta 2 líneas base + 3 modelos clásicos (SVM, Random Forest, XGBoost)
   sobre características visuales diseñadas (HSV + CIELAB + manchas + textura + HOG).
3. Entrena y ajusta 2 modelos de Deep Learning (CNN simple desde cero y MobileNetV2
   congelada con cabeza densa propia) registrando curvas por época.
4. Selecciona el modelo final únicamente por ``val_macro_f1`` (sin mirar ``test``).
5. Evalúa todos los modelos una sola vez sobre ``test``, genera métricas desglosadas
   por origen/especie, mide latencia y tamaño en disco (PI2), evalúa el estimador de
   tamaño sobre las 204 referencias y genera las figuras vectoriales (``.svg``/``.png``).
6. Exporta el modelo ganador listo para usar en ``06_despliegue/models/``.

Uso:
    .venv/bin/python 04_modelado/entrenar_modelos.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNOTATIONS_DIR,
    DIR_FIGURAS,
    DIR_METRICAS,
    DIR_MODELOS,
    FRUITNET_SPLITS_CSV,
    MIXED_STAGING_DIR,
    QUALITY_CLASSES,
    RANDOM_SEED,
)
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrices,
    plot_data_distribution,
    plot_dl_learning_curves,
    plot_error_analysis_grid,
    plot_model_comparison,
    plot_size_evaluation,
    save_dual_figure,
)
from src.features import extract_classical_features_rgb, extract_feature_matrix
from src.models_dl import (
    predict_dl_dataset,
    predict_single_pil,
    preload_cropped_images,
    save_dl_checkpoint,
    train_deep_learning_suite,
)
from src.models_ml import encode_labels, train_classical_suite
from src.preprocessing import load_clean_rgb
from src.size_estimation import estimate_fruit_size

SIZE_REFERENCES_CSV = ANNOTATIONS_DIR / "size_references.csv"


def _benchmark_ml_latency(estimator: Any, sample_paths: list[Path]) -> float:
    """Mide latencia promedio de inferencia por imagen (ms) incluyendo extracción clásica."""
    t0 = time.perf_counter()
    for p in sample_paths:
        rgb = np.asarray(load_clean_rgb(p))
        feat = extract_classical_features_rgb(rgb).reshape(1, -1)
        estimator.predict(feat)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return round(elapsed_ms / max(1, len(sample_paths)), 2)


def _benchmark_dl_latency(model: torch.nn.Module, image_size: int, device: torch.device, sample_paths: list[Path]) -> float:
    """Mide latencia promedio de inferencia por imagen (ms) incluyendo recorte y forward."""
    # Calentamiento breve
    _ = predict_single_pil(model, load_clean_rgb(sample_paths[0]), image_size=image_size, device=device)
    t0 = time.perf_counter()
    for p in sample_paths:
        _ = predict_single_pil(model, load_clean_rgb(p), image_size=image_size, device=device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return round(elapsed_ms / max(1, len(sample_paths)), 2)


def _breakdown_by_column(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column: str,
) -> dict[str, dict[str, Any]]:
    """Calcula exactitud y Macro-F1 sobre clases presentes para cada grupo de ``column``."""
    out: dict[str, dict[str, Any]] = {}
    for value, grp in test_df.groupby(column):
        idx = grp.index.to_numpy()
        yt = y_true[idx]
        yp = y_pred[idx]
        present = sorted(set(yt.tolist()))
        from sklearn.metrics import accuracy_score, f1_score

        out[str(value)] = {
            "n": int(len(yt)),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1_present_classes": round(
                float(f1_score(yt, yp, labels=present, average="macro", zero_division=0)), 4
            ),
            "present_classes": [QUALITY_CLASSES[i] for i in present],
        }
    return out


def evaluate_size_module(splits_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evalúa el estimador de tamaño sobre las 204 referencias y cobertura en todo el dataset."""
    ref_df = pd.read_csv(SIZE_REFERENCES_CSV)
    rows: list[dict[str, object]] = []
    ref_failures = 0

    for _, row in ref_df.iterrows():
        img = load_clean_rgb(PROJECT_ROOT / str(row["image_path"]))
        rgb = np.asarray(img)
        try:
            est = estimate_fruit_size(rgb)
            err = abs(est.normalized_diameter - float(row["reference_normalized_diameter"]))
            rows.append(
                {
                    "image_path": row["image_path"],
                    "species": row["species"],
                    "reference_normalized_diameter": float(row["reference_normalized_diameter"]),
                    "estimated_normalized_diameter": round(est.normalized_diameter, 6),
                    "diagonal_ratio": round(est.diagonal_ratio, 6),
                    "size_category": est.size_category,
                    "touches_border": est.touches_border,
                    "abs_error": round(err, 6),
                }
            )
        except Exception:
            ref_failures += 1

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(DIR_METRICAS / "evaluacion_tamano.csv", index=False)

    # Comprobar cobertura de segmentación en todas las imágenes del dataset combinado
    cov_by_source: dict[str, dict[str, int]] = {}
    for source, grp in splits_df.groupby("source"):
        ok_cnt, fail_cnt, border_cnt = 0, 0, 0
        for rel_path in grp["image_path"]:
            rgb = np.asarray(load_clean_rgb(PROJECT_ROOT / str(rel_path)))
            try:
                est = estimate_fruit_size(rgb)
                ok_cnt += 1
                if est.touches_border:
                    border_cnt += 1
            except Exception:
                fail_cnt += 1
        cov_by_source[str(source)] = {
            "total": int(len(grp)),
            "ok": ok_cnt,
            "failed": fail_cnt,
            "touches_border": border_cnt,
        }

    per_species = {
        str(sp): {
            "n": int(len(grp)),
            "mae": round(float(grp["abs_error"].mean()), 5),
            "median_ae": round(float(grp["abs_error"].median()), 5),
            "rmse": round(float(np.sqrt(np.mean(grp["abs_error"] ** 2))), 5),
        }
        for sp, grp in eval_df.groupby("species")
    }

    summary = {
        "formula": "max(bbox_width, bbox_height) / max(image_width, image_height)",
        "rotation_invariant": True,
        "references_total": int(len(ref_df)),
        "references_evaluated": int(len(eval_df)),
        "references_failed": int(ref_failures),
        "coverage_rate": round(len(eval_df) / max(1, len(ref_df)), 4),
        "mae": round(float(eval_df["abs_error"].mean()), 5),
        "median_ae": round(float(eval_df["abs_error"].median()), 5),
        "rmse": round(float(np.sqrt(np.mean(eval_df["abs_error"] ** 2))), 5),
        "per_species": per_species,
        "combined_dataset_coverage": cov_by_source,
    }
    (DIR_METRICAS / "evaluacion_tamano_resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return eval_df, summary


def export_mixed_segmentation_figure(fig_dir: Path) -> None:
    """Guarda en ``05_evaluacion/figuras/`` la evidencia visual de segmentación Mixed."""
    preview_path = MIXED_STAGING_DIR / "_preview_segmentacion.jpg"
    if not preview_path.is_file():
        return
    img = Image.open(preview_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title("Segmentación automática multifruto (CIELAB + Otsu + Watershed) sobre Mixed Qualit_Fruits")
    ax.axis("off")
    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "07_segmentacion_mixed")


def main() -> None:
    DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FRUITNET_SPLITS_CSV)
    train_df = df[df["split"].eq("train")].reset_index(drop=True)
    val_df = df[df["split"].eq("val")].reset_index(drop=True)
    test_df = df[df["split"].eq("test")].reset_index(drop=True)

    print(
        f"Particiones cargadas -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
        flush=True,
    )

    y_train = encode_labels(train_df["quality"])
    y_val = encode_labels(val_df["quality"])
    y_test = encode_labels(test_df["quality"])
    groups_train = train_df["leakage_group"].astype(str).to_numpy()

    # Gráfica 1: distribución de datos
    plot_data_distribution(df, DIR_FIGURAS)

    # --- FASE 4A: Modelos clásicos + Línea base ---
    print("\n[1/4] Extrayendo características clásicas (HSV + CIELAB + manchas + textura + HOG)...", flush=True)
    t_feat = time.perf_counter()
    X_train = extract_feature_matrix(train_df)
    X_val = extract_feature_matrix(val_df)
    X_test = extract_feature_matrix(test_df)
    feat_time = round(time.perf_counter() - t_feat, 2)
    print(f"  Matriz clásica: train={X_train.shape}, val={X_val.shape}, test={X_test.shape} ({feat_time}s)", flush=True)

    print("\n[2/4] Entrenando líneas base y modelos de ML clásico con GridSearchCV...", flush=True)
    ml_results = train_classical_suite(X_train, y_train, groups_train, X_val, y_val, seed=RANDOM_SEED)

    # --- FASE 4B: Deep Learning (CNN desde cero + MobileNetV2 congelada) ---
    print("\n[3/4] Precargando imágenes recortadas y entrenando Deep Learning...", flush=True)
    train_imgs = preload_cropped_images(train_df, max_side=256, use_foreground_crop=True)
    val_imgs = preload_cropped_images(val_df, max_side=256, use_foreground_crop=True)
    test_imgs = preload_cropped_images(test_df, max_side=256, use_foreground_crop=True)
    dl_results = train_deep_learning_suite(train_imgs, y_train, val_imgs, y_val, seed=RANDOM_SEED)

    # --- FASE 5: Evaluación unificada en conjunto de prueba reservado ---
    print("\n[4/4] Evaluando todos los modelos en prueba reservada y generando figuras...", flush=True)
    latency_paths = [PROJECT_ROOT / str(p) for p in test_df["image_path"].iloc[:25]]

    summary_rows: list[dict[str, Any]] = []
    full_details: dict[str, Any] = {}
    test_predictions_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model_sizes_mb: dict[str, float] = {}

    # Evaluar modelos clásicos y líneas base
    for name, info in ml_results.items():
        est = info["estimator"]
        y_pred_test = est.predict(X_test)
        probs_test = est.predict_proba(X_test) if hasattr(est, "predict_proba") else np.eye(len(QUALITY_CLASSES))[y_pred_test]
        test_metrics = compute_classification_metrics(y_test, y_pred_test)
        val_metrics = compute_classification_metrics(y_val, est.predict(X_val))

        tmp_file = DIR_METRICAS / "_tmp_size.joblib"
        joblib.dump(est, tmp_file, compress=3)
        size_mb = round(tmp_file.stat().st_size / (1024 * 1024), 3)
        tmp_file.unlink()
        model_sizes_mb[name] = size_mb

        latency_ms = _benchmark_ml_latency(est, latency_paths) if info["family"] == "ml_clasico" else 0.1
        gap = round(info["train_macro_f1"] - test_metrics["macro_f1"], 4)

        summary_rows.append(
            {
                "model": name,
                "family": info["family"],
                "train_macro_f1": round(info["train_macro_f1"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "overfitting_gap_train_minus_test": gap,
                "train_time_sec": info["train_time_sec"],
                "inference_ms_per_img": latency_ms,
                "model_size_mb": size_mb,
            }
        )
        full_details[name] = {
            "family": info["family"],
            "best_params": info["best_params"],
            "cv_train_macro_f1": round(info["cv_train_macro_f1"], 4),
            "cv_val_macro_f1": round(info["cv_val_macro_f1"], 4),
            "train_macro_f1": round(info["train_macro_f1"], 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_by_source": _breakdown_by_column(test_df, y_test, y_pred_test, "source"),
            "test_by_species": _breakdown_by_column(test_df, y_test, y_pred_test, "species"),
            "overfitting_gap": gap,
            "train_time_sec": info["train_time_sec"],
            "inference_ms_per_img": latency_ms,
            "model_size_mb": size_mb,
        }
        test_predictions_by_model[name] = (y_pred_test, probs_test)

    # Evaluar modelos Deep Learning
    dl_histories: dict[str, list[dict[str, float]]] = {}
    for name, info in dl_results.items():
        model = info["model"]
        device = info["device"]
        img_size = info["image_size"]

        y_pred_val, _ = predict_dl_dataset(model, val_imgs, image_size=img_size, device=device)
        y_pred_test, probs_test = predict_dl_dataset(model, test_imgs, image_size=img_size, device=device)

        val_metrics = compute_classification_metrics(y_val, y_pred_val)
        test_metrics = compute_classification_metrics(y_test, y_pred_test)

        tmp_ckpt = DIR_METRICAS / "_tmp_dl.pt"
        save_dl_checkpoint(model, info["arch_name"], img_size, tmp_ckpt)
        model.to(device)
        size_mb = round(tmp_ckpt.stat().st_size / (1024 * 1024), 3)
        tmp_ckpt.unlink()
        model_sizes_mb[name] = size_mb

        latency_ms = _benchmark_dl_latency(model, img_size, device, latency_paths)
        gap = round(info["train_macro_f1"] - test_metrics["macro_f1"], 4)
        dl_histories[name] = info["history"]

        summary_rows.append(
            {
                "model": name,
                "family": info["family"],
                "train_macro_f1": round(info["train_macro_f1"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "overfitting_gap_train_minus_test": gap,
                "train_time_sec": info["train_time_sec"],
                "inference_ms_per_img": latency_ms,
                "model_size_mb": size_mb,
            }
        )
        full_details[name] = {
            "family": info["family"],
            "arch_name": info["arch_name"],
            "image_size": img_size,
            "best_epoch": info["best_epoch"],
            "best_params": info["best_params"],
            "tuning_trials": info["tuning_trials"],
            "params_info": info["params_info"],
            "train_macro_f1": round(info["train_macro_f1"], 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_by_source": _breakdown_by_column(test_df, y_test, y_pred_test, "source"),
            "test_by_species": _breakdown_by_column(test_df, y_test, y_pred_test, "species"),
            "overfitting_gap": gap,
            "train_time_sec": info["train_time_sec"],
            "inference_ms_per_img": latency_ms,
            "model_size_mb": size_mb,
            "history": info["history"],
        }
        test_predictions_by_model[name] = (y_pred_test, probs_test)

    summary_df = pd.DataFrame(summary_rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    summary_df.to_csv(DIR_METRICAS / "resumen_modelos.csv", index=False)

    # Selección estricta del ganador basada SOLO en validación (sin usar test)
    non_baseline = summary_df[summary_df["family"] != "baseline"]
    winner_name = str(non_baseline.iloc[0]["model"])
    winner_family = str(non_baseline.iloc[0]["family"])
    best_classical_name = str(summary_df[summary_df["family"].eq("ml_clasico")].iloc[0]["model"])

    print("\n=== TABLA COMPARATIVA FINAL ===")
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nModelo seleccionado por validación (val_macro_f1): {winner_name}", flush=True)

    # Exportar artefactos del modelo ganador en 06_despliegue/models/
    deploy_meta: dict[str, Any] = {
        "model_name": winner_name,
        "family": winner_family,
        "classes": list(QUALITY_CLASSES),
        "selected_by": "val_macro_f1",
        "val_macro_f1": float(non_baseline.iloc[0]["val_macro_f1"]),
        "test_macro_f1": float(non_baseline.iloc[0]["test_macro_f1"]),
        "test_accuracy": float(non_baseline.iloc[0]["test_accuracy"]),
        "test_balanced_accuracy": float(non_baseline.iloc[0]["test_balanced_accuracy"]),
        "inference_ms_per_img": float(non_baseline.iloc[0]["inference_ms_per_img"]),
        "trained_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Limpiar artefactos anteriores en 06_despliegue/models/
    for old_file in DIR_MODELOS.glob("best_quality_model.*"):
        old_file.unlink()

    if winner_family == "deep_learning":
        win_info = dl_results[winner_name]
        ckpt_path = DIR_MODELOS / "best_quality_model.pt"
        save_dl_checkpoint(win_info["model"], win_info["arch_name"], win_info["image_size"], ckpt_path)
        deploy_meta["artifact_file"] = ckpt_path.name
        deploy_meta["arch_name"] = win_info["arch_name"]
        deploy_meta["image_size"] = win_info["image_size"]
        deploy_meta["artifact_size_mb"] = round(ckpt_path.stat().st_size / (1024 * 1024), 3)
    else:
        win_info = ml_results[winner_name]
        joblib_path = DIR_MODELOS / "best_quality_model.joblib"
        joblib.dump(win_info["estimator"], joblib_path, compress=3)
        deploy_meta["artifact_file"] = joblib_path.name
        deploy_meta["artifact_size_mb"] = round(joblib_path.stat().st_size / (1024 * 1024), 3)

    (DIR_MODELOS / "model_metadata.json").write_text(
        json.dumps(deploy_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Guardar predicciones detalladas de prueba
    pred_table = test_df[["image_path", "species", "source", "quality"]].copy()
    for m_name, (yp, _) in test_predictions_by_model.items():
        pred_table[f"pred_{m_name}"] = [QUALITY_CLASSES[int(i)] for i in yp]
    pred_table.to_csv(DIR_METRICAS / "predicciones_test.csv", index=False)

    # Evaluar estimación de tamaño sobre las 204 referencias y cobertura total
    size_eval_df, size_summary = evaluate_size_module(df)
    print(
        f"\nEvaluación de tamaño: {size_summary['references_evaluated']}/{size_summary['references_total']} "
        f"MAE={size_summary['mae']:.4f}, Mediana={size_summary['median_ae']:.4f}, RMSE={size_summary['rmse']:.4f}",
        flush=True,
    )

    # Guardar reporte JSON maestro con hardware y versiones (evidencia PI2)
    master_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "dataset_summary": {
            "total_images": int(len(df)),
            "unique_leakage_groups": int(df["leakage_group"].nunique()),
            "splits": df["split"].value_counts().to_dict(),
            "classes": df["quality"].value_counts().to_dict(),
            "sources": df["source"].value_counts().to_dict(),
            "species": df["species"].value_counts().to_dict(),
        },
        "environment_hardware": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_cores": os.cpu_count(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sklearn": sklearn.__version__,
            "classical_feature_extraction_sec": feat_time,
        },
        "selected_model_by_val": winner_name,
        "deployment_metadata": deploy_meta,
        "models": full_details,
        "size_evaluation": size_summary,
    }
    (DIR_METRICAS / "resultados_completos.json").write_text(
        json.dumps(master_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Generar todas las figuras vectoriales (.svg y .png) ---
    plot_model_comparison(summary_df, DIR_FIGURAS)
    plot_dl_learning_curves(dl_histories, DIR_FIGURAS)
    plot_confusion_matrices(
        {
            best_classical_name: full_details[best_classical_name]["test_metrics"]["confusion_matrix"],
            "CNN simple (desde cero)": full_details["CNN simple (desde cero)"]["test_metrics"]["confusion_matrix"],
            "MobileNetV2 (Transfer)": full_details["MobileNetV2 (Transfer)"]["test_metrics"]["confusion_matrix"],
        },
        DIR_FIGURAS,
    )
    win_yp, win_probs = test_predictions_by_model[winner_name]
    plot_error_analysis_grid(test_df, y_test, win_yp, win_probs, winner_name, DIR_FIGURAS)

    # 4 ejemplos diversos para la figura de tamaño (incluyendo fondo oscuro banano/manzana y tela clara)
    sample_size_imgs: list[tuple[str, Path]] = []
    for sp in ("banano", "manzana", "mango", "limon"):
        sub = df[(df["source"].eq("Propia")) & (df["species"].eq(sp))]
        if not sub.empty:
            sample_size_imgs.append((sp, PROJECT_ROOT / str(sub.iloc[0]["image_path"])))
    plot_size_evaluation(size_eval_df, sample_size_imgs, DIR_FIGURAS)
    export_mixed_segmentation_figure(DIR_FIGURAS)

    print(f"\nEntrenamiento, evaluación y figuras completados en {DIR_METRICAS.relative_to(PROJECT_ROOT)} y {DIR_FIGURAS.relative_to(PROJECT_ROOT)}.")


if __name__ == "__main__":
    main()
