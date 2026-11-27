"""Pruebas unitarias del estimador de tamaño relativo invariante a rotación."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.size_estimation import classify_size_category, estimate_fruit_size


def test_size_estimator_accuracy_and_rotation_invariance() -> None:
    """El diámetro normalizado debe ser exacto e invariante a rotaciones de 90°."""
    landscape = np.full((200, 400, 3), 240, dtype=np.uint8)
    cv2.circle(landscape, (200, 100), 50, (230, 110, 25), -1)

    est_land = estimate_fruit_size(landscape)
    # Diámetro esperado: ~101 px / max(400, 200) = 0.2525
    assert est_land.normalized_diameter == pytest.approx(101.0 / 400.0, abs=0.025)
    assert not est_land.touches_border

    portrait = np.rot90(landscape)
    est_port = estimate_fruit_size(portrait)
    assert est_port.normalized_diameter == pytest.approx(est_land.normalized_diameter, abs=0.01)


def test_size_estimator_works_on_dark_gray_countertop() -> None:
    """Debe segmentar correctamente frutas coloreadas sobre mesón gris oscuro."""
    dark_bg = np.full((240, 240, 3), 85, dtype=np.uint8)
    cv2.ellipse(dark_bg, (120, 120), (45, 75), 0, 0, 360, (235, 200, 30), -1)

    est = estimate_fruit_size(dark_bg)
    assert est.normalized_diameter == pytest.approx(150.0 / 240.0, abs=0.04)
    assert est.size_category == classify_size_category(est.normalized_diameter)


def test_size_estimator_rejects_uniform_blank_image() -> None:
    """Una imagen completamente uniforme sin fruto debe lanzar ``ValueError``."""
    blank = np.full((120, 120, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_fruit_size(blank)


def test_own_size_annotations_are_complete_and_coherent() -> None:
    """Verifica cobertura 307/307 y coherencia de umbrales en CSV de anotaciones propias."""
    import pandas as pd
    from src.config import ANNOTATIONS_DIR, OWN_ANNOTATIONS_CSV

    own = pd.read_csv(OWN_ANNOTATIONS_CSV)
    refs = pd.read_csv(ANNOTATIONS_DIR / "size_references.csv")
    assert len(own) == 307
    assert len(refs) == 307
    assert set(own["image_path"]) == set(refs["image_path"])
    assert own["reference_normalized_diameter"].notna().all()
    assert own["size_category"].notna().all()
    for _, row in refs.iterrows():
        expected_cat = classify_size_category(float(row["reference_normalized_diameter"]))
        assert row["reference_size_category"] == expected_cat
    merged = own.merge(refs[["image_path", "reference_size_category"]], on="image_path")
    assert (merged["size_category"] == merged["reference_size_category"]).all()
