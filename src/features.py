"""Extracción de características clásicas orientadas a calidad visual de frutas.

A diferencia de descriptores puramente geométricos, la calidad superficial depende
de color, uniformidad cromática, manchas oscuras localizadas y textura de la cáscara.

Vector de características por imagen (408 dimensiones):
1. Histogramas normalizados en espacio HSV (3 canales × 16 bins = 48 dims):
   capturan tono (madurez), saturación (frescura) y brillo global.
2. Estadísticos perceptuales en espacio CIELAB (3 canales × 6 estadísticos = 18 dims):
   media, desviación estándar, asimetría (skewness) y percentiles 10, 50 y 90 de
   ``L_opencv``, ``a`` y ``b`` para detectar oscurecimiento y pérdida de viveza.
3. Indicadores matemáticos de manchas y deterioro superficial (6 dims):
   - Proporción de píxeles muy oscuros (``L_opencv < 60`` en escala OpenCV ``[0, 255]``,
     equivalente aproximado a ``L* < 23.5`` en escala CIE ``[0, 100]``).
   - Proporción de píxeles pardos/magullados (tono cálido oscuro con baja luminancia).
   - Proporción de anomalías oscuras locales: ``L_opencv < max(35, mediana - 1.5 * IQR)``
     en escala ``[0, 255]`` (distingue manchas puntuales de ``Regular`` frente a deterioro global).
   - Número normalizado de manchas conexas y área relativa de la mayor mancha.
   - Contraste intercuartílico de luminancia ``IQR(L_opencv) / 255``.
4. Descriptores de textura y rugosidad (12 dims):
   energía de gradiente Sobel (media, desviación, p90), varianza y cuantiles del
   Laplaciano (moho, arrugas, cicatrices) y desviación local de luminancia (bloques 8×8).
5. Descriptor HOG compacto sobre ``64×64`` en escala de grises (324 dims):
   celdas ``16×16``, bloques ``32×32``, paso ``16`` y ``9`` orientaciones
   (formulación L2-Hys basada en Dalal & Triggs, CVPR 2005; implementado con OpenCV/NumPy).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import CLASSICAL_IMAGE_SIZE, PROJECT_ROOT
from src.preprocessing import load_clean_rgb
from src.size_estimation import crop_fruit_foreground

_HOG_SIZE = (64, 64)


def _compute_hog_64(gray_image: np.ndarray, cell_size: int = 16, nbins: int = 9) -> np.ndarray:
    """Calcula un descriptor HOG compacto (celdas 16×16, bloques 2×2, 9 orientaciones).

    Implementación vectorizada independiente de la versión de OpenCV (compatible con 4.x y 5.x):
    - Gradientes Sobel :math:`G_x, G_y` → magnitud :math:`M = \\sqrt{G_x^2 + G_y^2}` y
      orientación no dirigida :math:`\\theta \\in [0, \\pi)`.
    - Histogramas ponderados por magnitud en 4×4 celdas de 16×16 píxeles.
    - Normalización L2-Hys por bloques solapados de 2×2 celdas (3×3 bloques × 4 × 9 = 324 dims).
    """
    gray = cv2.resize(gray_image, _HOG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    angle = np.mod(np.rad2deg(np.arctan2(gy, gx)), 180.0)

    n_cells_y = _HOG_SIZE[1] // cell_size
    n_cells_x = _HOG_SIZE[0] // cell_size
    bin_width = 180.0 / nbins
    bin_idx = np.clip((angle / bin_width).astype(np.int32), 0, nbins - 1)

    cell_hists = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    for cy in range(n_cells_y):
        y0, y1 = cy * cell_size, (cy + 1) * cell_size
        for cx in range(n_cells_x):
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_hists[cy, cx] = np.bincount(
                bin_idx[y0:y1, x0:x1].ravel(),
                weights=mag[y0:y1, x0:x1].ravel(),
                minlength=nbins,
            )

    # Normalización L2 por bloques de 2×2 celdas con paso de 1 celda
    blocks: list[np.ndarray] = []
    eps = 1e-6
    for by in range(n_cells_y - 1):
        for bx in range(n_cells_x - 1):
            block = cell_hists[by : by + 2, bx : bx + 2, :].ravel()
            norm = float(np.sqrt(np.dot(block, block) + eps))
            block = np.minimum(block / norm, 0.2)
            norm2 = float(np.sqrt(np.dot(block, block) + eps))
            blocks.append(block / norm2)
    return np.concatenate(blocks).astype(np.float32)


def _skewness(values: np.ndarray) -> float:
    """Calcula la asimetría estandarizada de tercer momento evitando división por cero."""
    centered = values - float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def extract_classical_features_rgb(
    image_rgb: np.ndarray,
    *,
    use_foreground_crop: bool = True,
) -> np.ndarray:
    """Extrae el vector de características clásicas desde una imagen RGB ``uint8``."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Se requiere una imagen RGB de 3 canales.")

    working = crop_fruit_foreground(image_rgb)[0] if use_foreground_crop else image_rgb
    resized = cv2.resize(
        working,
        (CLASSICAL_IMAGE_SIZE, CLASSICAL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # 1. Histogramas normalizados en HSV (48 dims)
    hsv_hists: list[np.ndarray] = []
    for channel, value_range in enumerate(((0, 180), (0, 256), (0, 256))):
        hist = cv2.calcHist([hsv], [channel], None, [16], list(value_range)).ravel()
        total = float(hist.sum())
        hsv_hists.append((hist / total) if total > 0 else hist)

    # 2. Momentos y cuantiles en espacio CIELAB (18 dims)
    lab_stats: list[float] = []
    for ch in range(3):
        vals = lab[:, :, ch].ravel() / 255.0
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        lab_stats.extend(
            [
                float(vals.mean()),
                float(vals.std()),
                _skewness(vals),
                float(p10),
                float(p50),
                float(p90),
            ]
        )

    # 3. Indicadores de manchas y deterioro superficial (6 dims)
    l_opencv = lab[:, :, 0]  # Escala OpenCV [0, 255] (L* = L_opencv * 100 / 255)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # L_opencv < 60 equivale aproximadamente a L* < 23.5 en CIE [0, 100]
    dark_ratio = float(np.mean(l_opencv < 60.0))
    # Píxeles pardo-oscuros (tonos cálidos H <= 28 con saturación y brillo intermedio-bajo)
    brown_mask = (hue <= 28) & (sat >= 35) & (val >= 25) & (val <= 155)
    brown_ratio = float(np.mean(brown_mask))

    q25, q50, q75 = np.percentile(l_opencv, [25, 50, 75])
    iqr = max(1.0, float(q75 - q25))
    local_dark_thresh = max(35.0, float(q50 - 1.5 * iqr))
    spot_mask = (l_opencv < local_dark_thresh).astype(np.uint8)
    spot_ratio = float(np.mean(spot_mask))

    num_spots, _, spot_stats, _ = cv2.connectedComponentsWithStats(spot_mask, connectivity=8)
    valid_areas = [
        int(spot_stats[i, cv2.CC_STAT_AREA])
        for i in range(1, num_spots)
        if int(spot_stats[i, cv2.CC_STAT_AREA]) >= 12
    ]
    spot_count_norm = min(1.0, len(valid_areas) / 20.0)
    largest_spot_ratio = (max(valid_areas) / float(l_opencv.size)) if valid_areas else 0.0
    iqr_norm = iqr / 255.0

    defect_feats = np.array(
        [dark_ratio, brown_ratio, spot_ratio, spot_count_norm, largest_spot_ratio, iqr_norm],
        dtype=np.float32,
    )

    # 4. Textura: gradientes Sobel, Laplaciano y varianza por bloques (12 dims)
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    lap = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3))

    # Varianza local en rejilla 8×8 de bloques de 16×16 píxeles
    blocks = gray_f.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(64, 256)
    block_stds = blocks.std(axis=1)
    block_means = blocks.mean(axis=1)

    texture_feats = np.array(
        [
            float(grad_mag.mean()),
            float(grad_mag.std()),
            float(np.percentile(grad_mag, 90)),
            float(lap.mean()),
            float(lap.std()),
            float(np.percentile(lap, 90)),
            float(np.percentile(lap, 95)),
            float(block_stds.mean()),
            float(block_stds.std()),
            float(block_stds.max()),
            float(block_means.std()),
            float(np.percentile(block_means, 90) - np.percentile(block_means, 10)),
        ],
        dtype=np.float32,
    )

    # 5. HOG compacto sobre 64×64 (324 dims)
    hog_vec = _compute_hog_64(gray)

    return np.concatenate(
        [
            *hsv_hists,
            np.asarray(lab_stats, dtype=np.float32),
            defect_feats,
            texture_feats,
            hog_vec,
        ]
    ).astype(np.float32)


def extract_feature_matrix(
    frame: pd.DataFrame,
    root: Path = PROJECT_ROOT,
    *,
    use_foreground_crop: bool = True,
) -> np.ndarray:
    """Extrae la matriz ``X`` para todas las filas de un DataFrame con columna ``image_path``."""
    rows: list[np.ndarray] = []
    for rel_path in frame["image_path"]:
        img = load_clean_rgb(root / str(rel_path))
        rows.append(
            extract_classical_features_rgb(np.asarray(img), use_foreground_crop=use_foreground_crop)
        )
    return np.vstack(rows)
