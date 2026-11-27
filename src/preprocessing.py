"""Funciones de preprocesamiento de imágenes, limpieza EXIF y segmentación multifruto.

Este módulo concentra las operaciones reproducibles de Fase 3 (Preparación de datos):
- Carga segura con corrección de orientación EXIF.
- Limpieza completa de metadatos EXIF (privacidad - PI1) y redimensionado con Lanczos.
- Segmentación clásica de instancias individuales en escenas multifruto
  (`Mixed Qualit_Fruits`) mediante distancia cromática en espacio CIELAB,
  umbralización automática de Otsu (1979) y separación de frutos en contacto con
  Watershed (Meyer, 1994 / implementación OpenCV, Apache-2.0) (PI3).

Referencias de terceros reutilizadas:
- OpenCV (``cv2``) para transformaciones de color, morfología y Watershed.
- Pillow (``PIL``) para orientación EXIF y remuestreo Lanczos.
- Esquema estándar de hashing perceptual por diferencias horizontales (``dHash``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import JPEG_QUALITY, MAX_WORKING_SIDE_PX


@dataclass(frozen=True)
class FruitCrop:
    """Representa una fruta individual segmentada dentro de una escena multifruto."""

    bbox_xyxy: tuple[int, int, int, int]
    crop_rgb: np.ndarray
    area_ratio: float
    solidity: float


def sha256_bytes(data: bytes) -> str:
    """Calcula el resumen criptográfico SHA-256 de una secuencia de bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Calcula el SHA-256 de un archivo leyendo por bloques para bajo consumo de RAM."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_clean_rgb(path: str | Path, max_side: int | None = None) -> Image.Image:
    """Abre una imagen, corrige su orientación EXIF, convierte a RGB y elimina metadatos.

    Parámetros
    ----------
    path : str | Path
        Ruta al archivo de imagen.
    max_side : int | None
        Si se indica, reduce proporcionalmente la imagen usando interpolación Lanczos
        para que su lado mayor no supere ``max_side`` píxeles.
    """
    with Image.open(path) as raw:
        upright = ImageOps.exif_transpose(raw).convert("RGB")
        # Crear una nueva instancia desde los bytes RGB elimina cualquier metadato EXIF/GPS.
        clean = Image.frombytes("RGB", upright.size, upright.tobytes())
    if max_side is not None and max(clean.size) > max_side:
        clean.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return clean


def save_clean_jpeg(
    image: Image.Image,
    destination: str | Path,
    *,
    max_side: int | None = MAX_WORKING_SIDE_PX,
    quality: int = JPEG_QUALITY,
) -> tuple[int, int]:
    """Guarda una imagen RGB sin metadatos EXIF en formato JPEG optimizado.

    Devuelve las dimensiones finales ``(width, height)`` guardadas en disco.
    """
    working = image.convert("RGB").copy()
    if max_side is not None and max(working.size) > max_side:
        working.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    clean = Image.frombytes("RGB", working.size, working.tobytes())
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    clean.save(dst, format="JPEG", quality=quality, optimize=True)
    return clean.size


def _border_background_distance_lab(
    image_rgb: np.ndarray,
    border_frac: float = 0.07,
    luminance_weight: float = 0.35,
) -> np.ndarray:
    """Calcula la distancia ponderada en espacio CIELAB respecto al fondo de borde.

    Matemáticamente, para cada píxel :math:`(x, y)` con vector CIELAB
    :math:`(L^*, a^*, b^*)`, se estima el color de fondo :math:`(\\tilde{L}, \\tilde{a}, \\tilde{b})`
    mediante la mediana de los píxeles del marco exterior (robusta a frutos
    ocasionales cercanos al borde) y se evalúa:

    .. math::
        d(x, y) = \\sqrt{w_L (L^* - \\tilde{L})^2 + (a^* - \\tilde{a})^2 + (b^* - \\tilde{b})^2}

    Ponderar :math:`w_L = 0.35 < 1` atenúa sombras proyectadas sobre pisos o mesas
    neutras preservando el contraste cromático de las frutas y zonas oscuras extensas.
    """
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    bw = max(4, int(round(w * border_frac)))
    bh = max(4, int(round(h * border_frac)))
    border_pixels = np.concatenate(
        [
            lab[:bh, :, :].reshape(-1, 3),
            lab[h - bh :, :, :].reshape(-1, 3),
            lab[bh : h - bh, :bw, :].reshape(-1, 3),
            lab[bh : h - bh, w - bw :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_median = np.median(border_pixels, axis=0)

    diff = lab - bg_median
    dist = np.sqrt(
        luminance_weight * (diff[:, :, 0] ** 2)
        + (diff[:, :, 1] ** 2)
        + (diff[:, :, 2] ** 2)
    )
    return dist


def segment_multi_fruit_instances(
    image_rgb: np.ndarray,
    *,
    min_area_ratio: float = 0.010,
    max_area_ratio: float = 0.35,
    min_side_px: int = 22,
    min_solidity: float = 0.72,
    max_aspect_ratio: float = 4.0,
    pad_ratio: float = 0.14,
    min_lab_threshold: float = 15.0,
) -> list[FruitCrop]:
    """Segmenta frutas individuales en una imagen multifruto (ej. ``Mixed Qualit_Fruits``).

    Pipeline clásico sin dependencias externas:
    1. Mapa de distancia ponderada en CIELAB frente al color mediano del borde.
    2. Recorte de cola superior (percentil 97 / tope 60) para que frutas muy oscuras
       no desplacen el umbral de Otsu y oculten frutas claras en la misma escena.
    3. Apertura/cierre morfológico y relleno de huecos internos.
    4. Transformada de distancia euclidiana + Watershed para separar frutos en contacto.
    5. Filtrado geométrico estricto (área, solidez convexa, bordes) para conservar
       recortes limpios de un solo fruto.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Se requiere una imagen RGB de 3 canales.")

    h, w = image_rgb.shape[:2]
    total_area = float(h * w)

    # 1. Distancia CIELAB respecto al fondo estimado en los bordes
    dist = _border_background_distance_lab(image_rgb)
    dist_blur = cv2.GaussianBlur(dist, (5, 5), 0)

    # Acotar la cola superior evita que un fruto negro/podrido suba demasiado el umbral Otsu
    clip_hi = min(60.0, float(np.percentile(dist_blur, 98)))
    if clip_hi < min_lab_threshold:
        return []

    dist_clipped = np.clip(dist_blur, 0.0, clip_hi)
    dist_u8 = np.clip((dist_clipped / clip_hi) * 255.0, 0, 255).astype(np.uint8)
    otsu_u8, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_lab = (otsu_u8 / 255.0) * clip_hi
    threshold = max(min_lab_threshold, min(otsu_lab, 28.0))
    binary = (dist_blur >= threshold).astype(np.uint8)

    # 2. Limpieza morfológica y relleno de huecos internos
    k_open = max(3, (int(round(min(h, w) * 0.015)) | 1))
    k_close = max(3, (int(round(min(h, w) * 0.022)) | 1))
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)

    # 3. Separación de frutos que se tocan mediante Distance Transform + Watershed
    dist_tf = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

    sure_fg = np.zeros_like(filled, dtype=np.uint8)
    peak_size = 5 if min(h, w) <= 320 else 9
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size))
    for cc_id in range(1, num_cc):
        area = int(cc_stats[cc_id, cv2.CC_STAT_AREA])
        if area < min_area_ratio * total_area * 0.5:
            continue
        comp_mask = cc_labels == cc_id
        comp_dist = np.where(comp_mask, dist_tf, 0.0)
        comp_max = float(comp_dist.max())
        if comp_max < 5.0:
            continue
        local_max = (comp_dist == cv2.dilate(comp_dist, peak_kernel)) & (comp_dist >= 0.45 * comp_max)
        if np.count_nonzero(local_max) > 0:
            seed_mask = cv2.dilate(local_max.astype(np.uint8), np.ones((3, 3), np.uint8))
            sure_fg |= seed_mask & filled
        else:
            sure_fg |= (comp_dist >= 0.55 * comp_max).astype(np.uint8)

    sure_bg = cv2.dilate(filled, np.ones((5, 5), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ws_markers = cv2.watershed(bgr, markers)

    # 4. Extracción y filtrado geométrico de cada instancia segmentada
    crops: list[FruitCrop] = []
    for label_id in range(2, int(ws_markers.max()) + 1):
        inst_mask = (ws_markers == label_id).astype(np.uint8)
        area = int(np.count_nonzero(inst_mask))
        area_ratio = area / total_area
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        inst_contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not inst_contours:
            continue
        main_contour = max(inst_contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(main_contour)
        if min(bw, bh) < min_side_px:
            continue

        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > max_aspect_ratio:
            continue

        hull = cv2.convexHull(main_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(cv2.contourArea(main_contour) / hull_area) if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        # Descartar fragmentos pegados a bordes exteriores (pies, cortinas, frutos cortados)
        touches_edge = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if touches_edge:
            continue

        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        crop_rgb = image_rgb[y1:y2, x1:x2].copy()
        crops.append(
            FruitCrop(
                bbox_xyxy=(x1, y1, x2, y2),
                crop_rgb=crop_rgb,
                area_ratio=area_ratio,
                solidity=solidity,
            )
        )

    # Ordenar de arriba hacia abajo y de izquierda a derecha para trazabilidad estable
    crops.sort(key=lambda item: (item.bbox_xyxy[1] // 20, item.bbox_xyxy[0]))
    return crops
