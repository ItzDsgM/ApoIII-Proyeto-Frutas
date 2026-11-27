"""Pruebas de extracción clásica, arquitecturas DL y carga de modelo desplegado."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.config import DIR_MODELOS, QUALITY_CLASSES
from src.features import extract_classical_features_rgb
from src.models_dl import SimpleFruitCNN, build_transfer_mobilenet_v2, load_dl_checkpoint, predict_single_pil


def test_classical_feature_vector_is_finite_and_consistent() -> None:
    """Verifica dimensión exacta (408) y ausencia de NaN/Inf en features clásicos."""
    img = np.full((160, 160, 3), 240, dtype=np.uint8)
    img[40:120, 40:120] = (210, 120, 30)
    vec = extract_classical_features_rgb(img)
    assert vec.shape == (408,)
    assert np.all(np.isfinite(vec))


def test_simple_cnn_forward_and_mobilenet_freeze() -> None:
    """Comprueba salidas de CNN desde cero y congelación real de MobileNetV2."""
    cnn = SimpleFruitCNN(num_classes=len(QUALITY_CLASSES))
    out = cnn(torch.zeros((2, 3, 128, 128)))
    assert out.shape == (2, len(QUALITY_CLASSES))

    mob = build_transfer_mobilenet_v2(num_classes=len(QUALITY_CLASSES), pretrained=False)
    assert all(not p.requires_grad for p in mob.features.parameters())
    assert any(p.requires_grad for p in mob.classifier.parameters())


def test_deployed_model_predicts_valid_class_and_probabilities() -> None:
    """Comprueba que el modelo exportado en ``06_despliegue/models/`` funciona offline."""
    ckpt = DIR_MODELOS / "best_quality_model.pt"
    if not ckpt.is_file():
        return
    model, meta, device = load_dl_checkpoint(ckpt)
    sample = Image.new("RGB", (200, 200), (245, 245, 245))
    label, probs = predict_single_pil(model, sample, image_size=meta["image_size"], device=device)
    assert label in QUALITY_CLASSES
    assert set(probs.keys()) == set(QUALITY_CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-4
