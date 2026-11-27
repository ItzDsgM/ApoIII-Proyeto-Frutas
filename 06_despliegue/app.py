"""Interfaz web Streamlit para clasificación de calidad visual y estimación de tamaño.

Fase 6 CRISP-DM (Despliegue — Enunciado sec. 1.4):
- Permite cargar una fotografía o capturarla en tiempo real con la cámara.
- Estima la clase de calidad (``Buena``, ``Regular``, ``Mala``) junto con las
  probabilidades de salida del modelo ganador seleccionado en validación.
- Estima el diámetro normalizado invariante a rotación ``max(bw, bh) / max(W, H)``,
  su categoría relativa (``Pequeño``, ``Mediano``, ``Grande``) y muestra la máscara
  de segmentación para inspección transparente.

Uso desde la raíz del proyecto:
    .venv/bin/streamlit run 06_despliegue/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIR_METRICAS, DIR_MODELOS, QUALITY_CLASSES, QUALITY_DEFINITIONS
from src.features import extract_classical_features_rgb
from src.models_dl import load_dl_checkpoint, predict_single_pil
from src.size_estimation import crop_fruit_foreground, estimate_fruit_size

SPECIES_OPTIONS = (
    "Automática / otra",
    "banano",
    "limon",
    "mango",
    "manzana",
    "naranja",
    "tomate_arbol",
)

# Notas de dominio transparentes derivadas de la auditoría por especie y origen
SPECIES_DOMAIN_NOTES: dict[str, tuple[str, str]] = {
    "tomate_arbol": (
        "warning",
        "Advertencia de dominio: el modelo solo observó ejemplares `Buena` de `tomate_arbol` en "
        "entrenamiento/prueba (0 `Regular` y 0 `Mala` propios). Las predicciones de defecto en esta "
        "especie dependen de transferencia visual desde otras frutas.",
    ),
    "manzana": (
        "info",
        "Nota de dominio: las fotos propias de `manzana` son solo `Buena`; los ejemplos `Mala` y "
        "`Regular` provienen de Kaggle/Mixed. En fondos oscuros conviene validar visualmente golpes leves.",
    ),
    "mango": (
        "info",
        "Nota de dominio: `mango` no forma parte de Kaggle FruitNet; todas sus muestras son propias "
        "(58 en total, 7 en prueba). Se sugiere inspección humana ante manchas limítrofes.",
    ),
    "banano": (
        "info",
        "Nota de dominio: en prueba sobre fotos propias se detectaron falsos positivos `Mala → Buena` "
        "en bananos sobre fondo oscuro o con pardeamiento parcial. Verificar probabilidad.",
    ),
    "naranja": (
        "info",
        "Nota de dominio: los falsos positivos en `naranja` propia ocurrieron principalmente con daño "
        "localizado sobre fondo neutro/oscuro. Si la confianza es < 75 %, revisar manualmente.",
    ),
    "limon": (
        "info",
        "Nota de dominio: especie presente en fotos propias, Kaggle y Mixed; verifica iluminación "
        "uniforme para distinguir maduración normal de cicatrices superficiales.",
    ),
}

LOW_CONFIDENCE_THRESHOLD = 0.75


@st.cache_resource(show_spinner=False)
def load_evaluation_summary(metrics_dir: Path = DIR_METRICAS) -> dict[str, Any] | None:
    """Carga métricas de evaluación por especie para contextualizar la interfaz."""
    report_path = metrics_dir / "resultados_completos.json"
    if not report_path.is_file():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner=False)
def load_deployed_model(models_dir: Path = DIR_MODELOS) -> dict[str, Any] | None:
    """Carga y almacena en caché el modelo ganador exportado en ``06_despliegue/models/``."""
    meta_path = models_dir / "model_metadata.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifact_path = models_dir / str(meta["artifact_file"])
    if not artifact_path.is_file():
        return None

    if meta["family"] == "deep_learning":
        model, ckpt_meta, device = load_dl_checkpoint(artifact_path)
        return {
            "meta": meta,
            "family": "deep_learning",
            "model": model,
            "image_size": int(ckpt_meta["image_size"]),
            "device": device,
        }

    estimator = joblib.load(artifact_path)
    return {
        "meta": meta,
        "family": "ml_clasico",
        "estimator": estimator,
    }


def predict_quality(image_pil: Image.Image, bundle: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Ejecuta inferencia de calidad sobre una imagen RGB ya orientada."""
    if bundle["family"] == "deep_learning":
        # Igual que en la evaluación: primero recortar y luego reducir el lado mayor.
        cropped, _ = crop_fruit_foreground(np.asarray(image_pil.convert("RGB")))
        prepared = Image.fromarray(cropped)
        prepared.thumbnail((256, 256), Image.Resampling.LANCZOS)
        return predict_single_pil(
            bundle["model"],
            prepared,
            image_size=bundle["image_size"],
            device=bundle["device"],
            use_foreground_crop=False,
        )

    rgb = np.asarray(image_pil.convert("RGB"))
    features = extract_classical_features_rgb(rgb, use_foreground_crop=True).reshape(1, -1)
    est = bundle["estimator"]
    if hasattr(est, "predict_proba"):
        probs = est.predict_proba(features)[0]
    else:
        pred_idx = int(est.predict(features)[0])
        probs = np.eye(len(QUALITY_CLASSES))[pred_idx]
    pred_idx = int(np.argmax(probs))
    return QUALITY_CLASSES[pred_idx], {cls: float(probs[i]) for i, cls in enumerate(QUALITY_CLASSES)}


def draw_bbox_overlay(image_rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """Superpone la caja delimitadora y un tinte sutil sobre la máscara detectada."""
    overlay = image_rgb.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 220
    fg = mask > 0
    overlay[fg] = (0.80 * overlay[fg] + 0.20 * tint[fg]).astype(np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    thickness = max(2, min(overlay.shape[:2]) // 150)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 210, 80), thickness)
    return overlay


def inspect_fruit_image(
    image: Image.Image,
    models_dir: Path = DIR_MODELOS,
    selected_species: str = "Automática / otra",
) -> None:
    """Procesa y muestra en pantalla calidad visual, tamaño relativo y segmentación."""
    upright = ImageOps.exif_transpose(image).convert("RGB")
    working = upright.copy()
    working.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    rgb = np.asarray(working)

    t0 = time.perf_counter()
    bundle = load_deployed_model(models_dir)

    quality_label: str | None = None
    prob_map: dict[str, float] | None = None
    if bundle is not None:
        quality_label, prob_map = predict_quality(working, bundle)

    size_est = None
    size_error: str | None = None
    try:
        size_est = estimate_fruit_size(rgb)
    except (ValueError, cv2.error) as exc:
        size_error = str(exc)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # --- Panel superior de métricas clave ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calidad visual", quality_label or "Sin modelo")
    with col2:
        if prob_map is not None and quality_label is not None:
            st.metric("Puntuación del modelo", f"{prob_map[quality_label]:.1%}")
        else:
            st.metric("Puntuación del modelo", "N/A")
    with col3:
        if size_est is not None:
            st.metric("Tamaño relativo", f"{size_est.size_category} ({size_est.normalized_diameter:.3f})")
        else:
            st.metric("Tamaño relativo", "No estimado")
    with col4:
        st.metric("Latencia", f"{latency_ms:.0f} ms")

    if prob_map is not None and quality_label is not None and prob_map[quality_label] < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"Confianza moderada/baja ({prob_map[quality_label]:.1%} < {LOW_CONFIDENCE_THRESHOLD:.0%}). "
            "Según el protocolo *human-in-the-loop*, se recomienda revisión visual humana."
        )

    if selected_species != "Automática / otra":
        st.caption(f"Especie seleccionada para referencia contextual: **{selected_species}**")

    # --- Distribución de puntuaciones por clase ---
    if prob_map is not None:
        prob_df = pd.DataFrame(
            {"Clase": list(prob_map.keys()), "Puntuación": [round(v, 4) for v in prob_map.values()]}
        ).set_index("Clase")
        st.caption("Puntuaciones no calibradas: no equivalen a certeza ni a inocuidad.")
        st.bar_chart(prob_df, height=190)
    else:
        st.warning(
            "No se encontró un modelo entrenado en `06_despliegue/models/`. "
            "Ejecuta `python 04_modelado/entrenar_modelos.py` para generarlo, o instala el checkpoint verificado del equipo."
        )

    # --- Comparación visual: imagen original vs. segmentación de tamaño ---
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(working, caption=f"Imagen analizada ({working.width} × {working.height} px)", width="stretch")
    with img_col2:
        if size_est is not None:
            overlay = draw_bbox_overlay(rgb, size_est.bbox_xyxy, size_est.mask)
            st.image(
                overlay,
                caption=(
                    f"Diámetro normalizado: {size_est.normalized_diameter:.3f} "
                    f"(diagonal: {size_est.diagonal_ratio:.3f} | ocupación caja: {size_est.bbox_fill_ratio:.2f})"
                ),
                width="stretch",
            )
            if size_est.touches_border:
                st.info(
                    "Nota geométrica: el contorno toca el borde de la imagen; para mayor precisión "
                    "de tamaño, encuadra el fruto completo con margen alrededor."
                )
        else:
            st.warning(f"No se pudo segmentar el contorno del fruto: {size_error}")


def render() -> None:
    """Construye la aplicación principal de Streamlit."""
    st.set_page_config(
        page_title="Clasificador de Calidad y Tamaño de Frutas",
        page_icon="🍊",
        layout="wide",
    )
    st.title("Clasificación de Calidad Visual y Tamaño Relativo de Frutas")
    st.caption(
        "Proyecto Final — Algoritmos y Programación III (Universidad Icesi, 2026-2). "
        "Prototipo académico de apoyo visual; no reemplaza inspección sanitaria ni mide centímetros físicos."
    )

    bundle = load_deployed_model()
    eval_summary = load_evaluation_summary()
    with st.sidebar:
        st.header("Configuración e información")
        selected_species = st.selectbox("Especie de referencia (opcional)", SPECIES_OPTIONS)
        if bundle is not None:
            meta = bundle["meta"]
            st.success(f"Modelo activo: **{meta['model_name']}**")
            st.write(
                f"- **Macro-F1 prueba:** `{meta['test_macro_f1']:.4f}`\n"
                f"- **Exactitud prueba:** `{meta['test_accuracy']:.4f}`\n"
                f"- **Tamaño en disco:** `{meta['artifact_size_mb']:.2f} MB`"
            )
        if selected_species != "Automática / otra":
            st.divider()
            st.subheader(f"Referencia: `{selected_species}`")
            if eval_summary is not None:
                model_key = bundle["meta"]["model_name"] if bundle is not None else eval_summary.get("selected_model_by_val")
                sp_quality = eval_summary.get("models", {}).get(model_key, {}).get("test_by_species", {}).get(selected_species)
                sp_size = eval_summary.get("size_evaluation", {}).get("per_species", {}).get(selected_species)
                lines: list[str] = []
                if sp_quality:
                    lines.append(
                        f"- **Calidad (test, n={sp_quality['n']}):** Acc `{sp_quality['accuracy']:.2%}` | "
                        f"Macro-F1 `{sp_quality['macro_f1_present_classes']:.4f}`"
                    )
                if sp_size:
                    lines.append(
                        f"- **Tamaño (refs, n={sp_size['n']}):** MAE `{sp_size['mae']:.4f}` | "
                        f"Acc cat. `{sp_size.get('category_accuracy', 0.0):.2%}`"
                    )
                if lines:
                    st.markdown("\n".join(lines))
            note = SPECIES_DOMAIN_NOTES.get(selected_species)
            if note is not None:
                level, text = note
                if level == "warning":
                    st.warning(text)
                else:
                    st.info(text)
        st.divider()
        st.subheader("Criterios de calidad")
        for label, definition in QUALITY_DEFINITIONS.items():
            st.markdown(f"**{label}:** {definition}")
        st.divider()
        st.caption(
            "Diámetro normalizado = `max(ancho_caja, alto_caja) / max(W, H)` "
            "(invariante a rotaciones de 90°). Las imágenes se procesan localmente sin enviarse a servidores externos."
        )

    source = st.radio(
        "Modo de entrada",
        ("Cargar fotografía", "Capturar con cámara"),
        horizontal=True,
    )
    uploaded = (
        st.camera_input("Captura una sola fruta sobre fondo sencillo")
        if source == "Capturar con cámara"
        else st.file_uploader("Sube una imagen de una fruta individual", type=["jpg", "jpeg", "png", "webp"])
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            image.load()
            inspect_fruit_image(image, selected_species=selected_species)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("No se pudo leer el archivo como una imagen válida.")


if __name__ == "__main__":
    render()
