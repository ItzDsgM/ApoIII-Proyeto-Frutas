"""Pruebas unitarias del módulo de preprocesamiento y segmentación multifruto."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.preprocessing import (
    load_clean_rgb,
    save_clean_jpeg,
    segment_multi_fruit_instances,
)


def test_save_clean_jpeg_strips_exif_and_bounds_resolution(tmp_path) -> None:
    """Verifica que guardar JPEG elimina metadatos EXIF y respeta ``max_side``."""
    raw_img = Image.new("RGB", (1600, 800), (220, 120, 40))
    exif = raw_img.getexif()
    exif[271] = "TestPhone"
    src_path = tmp_path / "with_exif.jpg"
    raw_img.save(src_path, format="JPEG", exif=exif)

    loaded = load_clean_rgb(src_path, max_side=800)
    assert loaded.size == (800, 400)
    assert len(loaded.getexif()) == 0

    dst_path = tmp_path / "clean.jpg"
    saved_size = save_clean_jpeg(loaded, dst_path, max_side=500)
    assert saved_size == (500, 250)
    with Image.open(dst_path) as reloaded:
        assert reloaded.size == (500, 250)
        assert len(reloaded.getexif()) == 0


def test_segment_multi_fruit_separates_multiple_fruits() -> None:
    """Comprueba la detección de varias frutas sintéticas bien separadas sobre fondo neutro."""
    scene = np.full((240, 320, 3), 235, dtype=np.uint8)
    cv2.circle(scene, (75, 120), 32, (235, 130, 20), -1)
    cv2.circle(scene, (175, 115), 30, (210, 45, 35), -1)
    cv2.circle(scene, (260, 135), 26, (220, 210, 30), -1)

    crops = segment_multi_fruit_instances(scene)
    assert len(crops) == 3
    for crop in crops:
        assert crop.crop_rgb.ndim == 3
        assert min(crop.crop_rgb.shape[:2]) >= 22
