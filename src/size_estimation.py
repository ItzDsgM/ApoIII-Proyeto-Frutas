"""Estimación de tamaño relativo de frutas invariante a orientación.

Implementa la segunda salida requerida por el enunciado (sección 1.3):
- Diámetro normalizado en píxeles: ``max(bw, bh) / max(W, H)`` en el rango ``(0, 1]``,
  invariante a rotaciones de 90° (a diferencia de dividir solo por el ancho ``W``).
- Categoría relativa de tamaño (``Pequeño``, ``Mediano``, ``Grande``) calculada a
  partir de umbrales calibrables.

Fundamento matemático (PI3):
Se combina la distancia cromática ponderada en CIELAB frente al fondo de los bordes
(``w_L = 0.30`` para atenuar sombras proyectadas) con el canal de saturación HSV
cuando el fondo es neutro (blanco, gris claro o mesón gris oscuro), seguida de
umbralización automática de Otsu, morfología matemática y selección ponderada por
centralidad para aislar el fruto principal incluso si aparecen bordes secundarios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import SIZE_CATEGORIES
from src.preprocessing import _border_background_distance_lab

# Umbrales por defecto para clasificar el diámetro normalizado (respecto al lado mayor)
# en categorías relativas Pequeño / Mediano / Grande.
DEFAULT_SIZE_THRESHOLDS: tuple[float, float] = (0.30, 0.55)


@dataclass(frozen=True)
class SizeEstimate:
    """Resultado estructurado de la estimación de tamaño relativo."""

    normalized_diameter: float
    diagonal_ratio: float
    area_ratio: float
    size_category: str
    bbox_xyxy: tuple[int, int, int, int]
    bbox_fill_ratio: float
    touches_border: bool
    mask: np.ndarray


def classify_size_category(
    normalized_diameter: float,
    thresholds: tuple[float, float] = DEFAULT_SIZE_THRESHOLDS,
) -> str:
    """Asigna una categoría relativa (``Pequeño``, ``Mediano``, ``Grande``)."""
    low, high = thresholds
    if normalized_diameter < low:
        return SIZE_CATEGORIES[0]
    if normalized_diameter < high:
        return SIZE_CATEGORIES[1]
    return SIZE_CATEGORIES[2]


def estimate_fruit_size(
    image_rgb: np.ndarray,
    *,
    max_working_side: int = 800,
    thresholds: tuple[float, float] = DEFAULT_SIZE_THRESHOLDS,
) -> SizeEstimate:
    """Segmenta el fruto principal y estima su diámetro relativo normalizado.

    Parámetros
    ----------
    image_rgb : np.ndarray
        Imagen RGB ``uint8`` de forma ``(H, W, 3)``.
    max_working_side : int
        Lado mayor máximo para acelerar la segmentación conservando la proporción.
    thresholds : tuple[float, float]
        Cortes ``(t1, t2)`` para asignar ``Pequeño / Mediano / Grande``.

    Devuelve
    --------
    SizeEstimate
        Objeto con métricas geométricas normalizadas, caja delimitadora y máscara binaria
        a la resolución de entrada ``(H, W)``.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Se requiere una imagen RGB uint8 de forma (H, W, 3).")

    orig_h, orig_w = image_rgb.shape[:2]
    if min(orig_h, orig_w) < 32:
        raise ValueError("La imagen es demasiado pequeña para estimar el contorno del fruto.")

    # Escalar copia de trabajo para mantener latencia baja (~10-20 ms en CPU)
    scale = 1.0
    if max(orig_h, orig_w) > max_working_side:
        scale = max_working_side / float(max(orig_h, orig_w))
        work_w = max(32, int(round(orig_w * scale)))
        work_h = max(32, int(round(orig_h * scale)))
        work_rgb = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_rgb = image_rgb
        work_h, work_w = orig_h, orig_w

    hsv = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # Estimar saturación del fondo en el marco exterior (7 %)
    bw_strip = max(4, int(round(work_w * 0.07)))
    bh_strip = max(4, int(round(work_h * 0.07)))
    border_sat = np.concatenate(
        [
            sat[:bh_strip, :].ravel(),
            sat[work_h - bh_strip :, :].ravel(),
            sat[bh_strip : work_h - bh_strip, :bw_strip].ravel(),
            sat[bh_strip : work_h - bh_strip, work_w - bw_strip :].ravel(),
        ]
    )
    bg_sat_median = float(np.median(border_sat))

    lab_dist = _border_background_distance_lab(work_rgb, border_frac=0.07, luminance_weight=0.30)

    # Si el fondo es neutro (blanco, tela clara o mesón gris oscuro), combinar distancia Lab + saturación
    if bg_sat_median < 55.0:
        score_map = np.maximum(lab_dist, 0.55 * np.maximum(0.0, sat - bg_sat_median))
    else:
        score_map = lab_dist

    score_blur = cv2.GaussianBlur(score_map, (5, 5), 0)
    clip_hi = min(90.0, float(np.percentile(score_blur, 99)))
    if clip_hi < 14.0:
        raise ValueError("Contraste insuficiente entre la fruta y el fondo.")

    score_u8 = np.clip((np.minimum(score_blur, clip_hi) / clip_hi) * 255.0, 0, 255).astype(np.uint8)
    otsu_u8, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = (otsu_u8 / 255.0) * clip_hi
    threshold = max(14.0, min(otsu_val, 42.0))
    binary = (score_blur >= threshold).astype(np.uint8)

    # Cierre y apertura morfológica adaptados al tamaño de la imagen
    k_close = max(5, (int(round(min(work_h, work_w) * 0.025)) | 1))
    k_open = max(3, (int(round(min(work_h, work_w) * 0.012)) | 1))
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    )

    # Seleccionar el componente conexo principal ponderado por centralidad y área
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_area = float(work_h * work_w)
    best_score = -1.0
    best_id = -1

    for cc_id in range(1, num_cc):
        x, y, bw, bh, area = stats[cc_id]
        area_frac = area / total_area
        if area_frac < 0.008 or area_frac > 0.96:
            continue
        cx, cy = centroids[cc_id]
        dx = (cx - work_w / 2.0) / (work_w / 2.0)
        dy = (cy - work_h / 2.0) / (work_h / 2.0)
        center_weight = math.exp(-0.5 * ((dx**2 + dy**2) / (0.65**2)))
        # Penalizar ligeramente objetos pegados a un borde si existe un fruto central limpio
        touches_any = x <= 1 or y <= 1 or (x + bw) >= work_w - 1 or (y + bh) >= work_h - 1
        border_factor = 0.72 if touches_any and area_frac < 0.35 else 1.0
        candidate_score = area_frac * center_weight * border_factor
        if candidate_score > best_score:
            best_score = candidate_score
            best_id = cc_id

    if best_id < 0:
        raise ValueError("No se detectó un fruto claramente separado del fondo.")

    work_mask = (labels == best_id).astype(np.uint8)
    contours, _ = cv2.findContours(work_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_work = np.zeros_like(work_mask)
    cv2.drawContours(filled_work, contours, -1, 1, cv2.FILLED)

    # Reescalar máscara al tamaño original para devolver coordenadas exactas
    if (work_w, work_h) != (orig_w, orig_h):
        full_mask = cv2.resize(filled_work, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        full_mask = filled_work

    ys, xs = np.nonzero(full_mask)
    if len(xs) == 0:
        raise ValueError("Máscara vacía tras el posprocesamiento.")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = x2 - x1
    bh = y2 - y1

    max_img_side = float(max(orig_w, orig_h))
    diag_img = math.hypot(orig_w, orig_h)
    normalized_diameter = float(max(bw, bh) / max_img_side)
    diagonal_ratio = float(max(bw, bh) / diag_img)
    mask_area = float(np.count_nonzero(full_mask))
    area_ratio = float(mask_area / (orig_w * orig_h))
    bbox_fill_ratio = float(mask_area / max(1, bw * bh))
    touches_border = bool(x1 <= 2 or y1 <= 2 or x2 >= orig_w - 2 or y2 >= orig_h - 2)

    return SizeEstimate(
        normalized_diameter=normalized_diameter,
        diagonal_ratio=diagonal_ratio,
        area_ratio=area_ratio,
        size_category=classify_size_category(normalized_diameter, thresholds=thresholds),
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_fill_ratio=bbox_fill_ratio,
        touches_border=touches_border,
        mask=full_mask,
    )


def crop_fruit_foreground(
    image_rgb: np.ndarray,
    *,
    pad_ratio: float = 0.08,
) -> tuple[np.ndarray, SizeEstimate | None]:
    """Recorta ajustadamente el fruto detectado para la clasificación de calidad.

    A diferencia del cálculo estricto de tamaño, si el fruto ocupa casi toda la imagen
    o la segmentación no logra aislarlo, conserva la imagen completa de forma segura.
    """
    try:
        est = estimate_fruit_size(image_rgb)
    except (ValueError, cv2.error):
        return image_rgb, None

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = est.bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return image_rgb[cy1:cy2, cx1:cx2].copy(), est
