"""Métricas cuantitativas y generación de figuras vectoriales (Fase 5 CRISP-DM).

Produce evidencias directamente alineadas con la Rúbrica (Criterios 5, 6 y 8) y PI2/PI3:
- Métricas de clasificación: Exactitud, Exactitud Balanceada, Macro-F1, Weighted-F1,
  precisión/sensibilidad/F1 por clase y matriz de confusión.
- Diagnóstico cuantitativo de sobreajuste (brecha ``train_macro_f1 - test_macro_f1``).
- Desglose de desempeño por origen (Propias vs. Kaggle) y por especie.
- Evaluación completa de estimación de tamaño (MAE, RMSE, mediana de error, cobertura).
- Gráficas vectoriales (``.svg``) y raster (``.png``) listas para el informe IEEE y video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import PROJECT_ROOT, QUALITY_CLASSES
from src.size_estimation import estimate_fruit_size

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calcula el conjunto completo de métricas sobre índices ``0..K-1``."""
    labels = list(range(len(QUALITY_CLASSES)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(QUALITY_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall": round(float(report[cls]["recall"]), 4),
            "f1": round(float(report[cls]["f1-score"]), 4),
            "support": int(report[cls]["support"]),
        }
        for cls in QUALITY_CLASSES
    }
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_dual_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Guarda cada gráfica en formato vectorial ``.svg`` (para informe IEEE) y ``.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.svg")
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    plt.close(fig)


def plot_data_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    """Gráfica 1: distribución de clases por fuente de datos y por especie."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ct_source = pd.crosstab(df["source"], df["quality"]).reindex(columns=list(QUALITY_CLASSES), fill_value=0)
    ct_source.plot(kind="bar", stacked=True, ax=axes[0], color=["#2E8B57", "#E6A12E", "#C0392B"])
    axes[0].set_title("Distribución de clases por origen")
    axes[0].set_xlabel("Origen de los datos")
    axes[0].set_ylabel("Número de imágenes")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend(title="Calidad")

    ct_species = pd.crosstab(df["species"], df["quality"]).reindex(columns=list(QUALITY_CLASSES), fill_value=0)
    ct_species.plot(kind="bar", stacked=True, ax=axes[1], color=["#2E8B57", "#E6A12E", "#C0392B"])
    axes[1].set_title("Distribución de clases por especie")
    axes[1].set_xlabel("Especie")
    axes[1].set_ylabel("Número de imágenes")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="Calidad")

    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "01_distribucion_datos")


def plot_model_comparison(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    """Gráfica 2: comparativa Train vs Val vs Test Macro-F1 (diagnóstico de sobreajuste)."""
    ordered = summary_df.sort_values("test_macro_f1", ascending=True)
    y_pos = np.arange(len(ordered))
    height = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(y_pos + height, ordered["train_macro_f1"], height, label="Train Macro-F1", color="#90CAF9")
    ax.barh(y_pos, ordered["val_macro_f1"], height, label="Val Macro-F1", color="#42A5F5")
    ax.barh(y_pos - height, ordered["test_macro_f1"], height, label="Test Macro-F1", color="#1565C0")

    for i, (_, row) in enumerate(ordered.iterrows()):
        ax.text(float(row["test_macro_f1"]) + 0.012, i - height, f"{row['test_macro_f1']:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro-F1 (3 clases)")
    ax.set_title("Comparativa de modelos y brecha Train / Val / Test (sobreajuste)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "02_comparativa_modelos")


def plot_dl_learning_curves(dl_histories: dict[str, list[dict[str, float]]], fig_dir: Path) -> None:
    """Gráfica 3: curvas de pérdida y Macro-F1 por época (CNN desde cero vs. MobileNetV2)."""
    names = list(dl_histories.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.8 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])

    for row_idx, name in enumerate(names):
        hist = pd.DataFrame(dl_histories[name])
        ax_loss, ax_f1 = axes[row_idx]

        ax_loss.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train Loss", color="#1E88E5")
        ax_loss.plot(hist["epoch"], hist["val_loss"], marker="s", label="Val Loss", color="#E53935")
        ax_loss.set_title(f"{name} — Pérdida (Cross-Entropy ponderada)")
        ax_loss.set_xlabel("Época")
        ax_loss.set_ylabel("Pérdida")
        ax_loss.grid(True, linestyle="--", alpha=0.35)
        ax_loss.legend()

        ax_f1.plot(hist["epoch"], hist["train_macro_f1"], marker="o", label="Train Macro-F1", color="#2E7D32")
        ax_f1.plot(hist["epoch"], hist["val_macro_f1"], marker="s", label="Val Macro-F1", color="#FB8C00")
        ax_f1.set_title(f"{name} — Macro-F1 por época")
        ax_f1.set_xlabel("Época")
        ax_f1.set_ylabel("Macro-F1")
        ax_f1.set_ylim(0, 1.05)
        ax_f1.grid(True, linestyle="--", alpha=0.35)
        ax_f1.legend()

    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "03_curvas_aprendizaje_dl")


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    """Gráfica 4: matrices de confusión (conteo y porcentaje por fila) en conjunto de prueba."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, matrices.items()):
        cm_arr = np.asarray(cm, dtype=np.int64)
        row_sums = np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)
        cm_pct = cm_arr / row_sums
        annot = np.empty_like(cm_arr, dtype=object)
        for r in range(cm_arr.shape[0]):
            for c in range(cm_arr.shape[1]):
                annot[r, c] = f"{cm_arr[r, c]}\n({cm_pct[r, c] * 100:.0f}%)"
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=list(QUALITY_CLASSES),
            yticklabels=list(QUALITY_CLASSES),
            cbar=False,
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")

    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "04_matrices_confusion")


def plot_error_analysis_grid(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    fig_dir: Path,
    root: Path = PROJECT_ROOT,
) -> None:
    """Gráfica 5: galería visual de errores reales y aciertos en el conjunto de prueba."""
    wrong_idx = np.flatnonzero(y_true != y_pred)
    right_idx = np.flatnonzero(y_true == y_pred)

    # Priorizar mostrar hasta 6 errores reales + 2 aciertos representativos (8 paneles)
    chosen = list(wrong_idx[:6])
    need_right = max(0, 8 - len(chosen))
    if len(right_idx) > 0 and need_right > 0:
        step = max(1, len(right_idx) // need_right)
        chosen.extend(right_idx[::step][:need_right].tolist())

    if not chosen:
        return

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.3 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, idx in zip(axes_flat, chosen):
        row = test_df.iloc[int(idx)]
        img = Image.open(root / str(row["image_path"])).convert("RGB")
        ax.imshow(img)
        true_lbl = QUALITY_CLASSES[int(y_true[idx])]
        pred_lbl = QUALITY_CLASSES[int(y_pred[idx])]
        conf = float(probs[idx, int(y_pred[idx])])
        ok = true_lbl == pred_lbl
        color = "#1B5E20" if ok else "#B71C1C"
        tag = "ACIERTO" if ok else "ERROR"
        ax.set_title(
            f"[{tag}] {row['species']}\nReal: {true_lbl} | Pred: {pred_lbl} ({conf:.0%})",
            color=color,
            fontsize=8.5,
        )

    fig.suptitle(f"Análisis visual de predicciones en prueba — {model_name}", fontsize=12, y=1.02)
    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "05_analisis_errores_visual")


def plot_size_evaluation(
    eval_df: pd.DataFrame,
    sample_paths: list[tuple[str, Path]],
    fig_dir: Path,
) -> None:
    """Gráfica 6: dispersión estimado vs. referencia (307 cajas) + distribución de error + máscaras."""
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0])

    ax_scatter = fig.add_subplot(gs[0, :2])
    for sp, grp in eval_df.groupby("species"):
        ax_scatter.scatter(
            grp["reference_normalized_diameter"],
            grp["estimated_normalized_diameter"],
            alpha=0.75,
            s=28,
            label=f"{sp} (MAE={grp['abs_error'].mean():.3f})",
        )
    lim_max = max(0.6, float(eval_df[["reference_normalized_diameter", "estimated_normalized_diameter"]].max().max()) + 0.04)
    ax_scatter.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="Ideal (y = x)")
    ax_scatter.set_xlabel("Diámetro normalizado de referencia")
    ax_scatter.set_ylabel("Diámetro normalizado estimado")
    ax_scatter.set_title(f"Tamaño estimado vs. referencia (n={len(eval_df)}, MAE={eval_df['abs_error'].mean():.4f})")
    ax_scatter.grid(True, linestyle="--", alpha=0.35)
    ax_scatter.legend(fontsize=8)

    ax_hist = fig.add_subplot(gs[0, 2:])
    ax_hist.hist(eval_df["abs_error"], bins=24, color="#26A69A", edgecolor="white", alpha=0.88)
    ax_hist.axvline(eval_df["abs_error"].mean(), color="#D32F2F", linestyle="--", label=f"MAE = {eval_df['abs_error'].mean():.4f}")
    ax_hist.axvline(eval_df["abs_error"].median(), color="#1565C0", linestyle=":", label=f"Mediana = {eval_df['abs_error'].median():.4f}")
    ax_hist.set_xlabel("Error absoluto |estimado - referencia|")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.set_title("Distribución del error absoluto de tamaño")
    ax_hist.grid(True, linestyle="--", alpha=0.35)
    ax_hist.legend(fontsize=8.5)

    for col_idx, (title, img_path) in enumerate(sample_paths[:4]):
        ax_img = fig.add_subplot(gs[1, col_idx])
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        est = estimate_fruit_size(rgb)
        overlay = rgb.copy()
        x1, y1, x2, y2 = est.bbox_xyxy
        # Resaltar contorno de la caja en verde brillante
        import cv2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 230, 64), max(2, min(rgb.shape[:2]) // 120))
        ax_img.imshow(overlay)
        ax_img.set_title(f"{title}\nd={est.normalized_diameter:.3f} ({est.size_category})", fontsize=8.5)
        ax_img.axis("off")

    fig.tight_layout()
    save_dual_figure(fig, fig_dir, "06_evaluacion_tamano")
