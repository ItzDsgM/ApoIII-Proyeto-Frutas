"""Modelos de Deep Learning: CNN desde cero y Transfer Learning con MobileNetV2.

Implementa las dos alternativas recomendadas en la sección 3 del enunciado:
1. ``SimpleFruitCNN``: arquitectura convolucional liviana entrenada desde cero
   (3 bloques Conv-BatchNorm-ReLU-MaxPool + GlobalAvgPool + capas densas con Dropout).
2. ``TransferMobileNetV2``: MobileNetV2 preentrenada en ImageNet con el extractor
   congelado (incluyendo estadísticas BatchNorm en modo ``eval``) y cabeza densa propia.

Ambos modelos:
- Utilizan entropía cruzada ponderada por frecuencia inversa de clase
  (:math:`w_c = N / (K \\cdot N_c)`) y aumento de datos en entrenamiento.
- Registran pérdida, exactitud y Macro-F1 en entrenamiento y validación por época
  para diagnosticar sobreajuste y generalización (Rúbrica Criterio 6).
- Seleccionan hiperparámetros (tasa de aprendizaje) y mejor época únicamente con el
  conjunto de validación (``val``), sin consultar el conjunto de prueba (``test``).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.config import DL_IMAGE_SIZE, PROJECT_ROOT, QUALITY_CLASSES, QUALITY_TO_IDX, RANDOM_SEED
from src.preprocessing import load_clean_rgb
from src.size_estimation import crop_fruit_foreground

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCRATCH_CNN_SIZE = 128


def set_torch_seed(seed: int = RANDOM_SEED) -> None:
    """Fija semillas de NumPy y PyTorch para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int, *, training: bool) -> transforms.Compose:
    """Construye transformaciones con Data Augmentation moderado en entrenamiento."""
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if training:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.35),
                transforms.RandomRotation(degrees=18),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class PreloadedFruitDataset(Dataset):
    """Dataset en memoria con recorte del fruto precargado para acelerar épocas en CPU."""

    def __init__(
        self,
        images_pil: list[Image.Image],
        labels: np.ndarray,
        transform: transforms.Compose,
    ) -> None:
        self.images = images_pil
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tensor = self.transform(self.images[index])
        return tensor, int(self.labels[index])


def preload_cropped_images(
    frame: pd.DataFrame,
    root: Path = PROJECT_ROOT,
    *,
    max_side: int = 256,
    use_foreground_crop: bool = True,
) -> list[Image.Image]:
    """Carga, limpia y recorta en memoria las imágenes de un DataFrame."""
    loaded: list[Image.Image] = []
    for rel_path in frame["image_path"]:
        img = load_clean_rgb(root / str(rel_path))
        if use_foreground_crop:
            cropped_rgb, _ = crop_fruit_foreground(np.asarray(img))
            img = Image.fromarray(cropped_rgb)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        loaded.append(img)
    return loaded


class SimpleFruitCNN(nn.Module):
    """CNN pequeña de 3 bloques convolucionales entrenada desde cero."""

    def __init__(self, num_classes: int = len(QUALITY_CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Bloque 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Bloque 3: 32x32 -> 16x16 -> Global Average Pooling (1x1)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_transfer_mobilenet_v2(
    num_classes: int = len(QUALITY_CLASSES),
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Construye MobileNetV2 con extractor congelado y cabeza densa propia."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20),
        nn.Linear(128, num_classes),
    )
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Cuenta parámetros totales y entrenables de un modelo PyTorch."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa pérdida promedio, exactitud, Macro-F1 y probabilidades sobre un DataLoader."""
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_n = int(labels.size(0))
            loss_sum += float(loss.item()) * batch_n
            n_samples += batch_n
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

    probs_arr = np.vstack(all_probs)
    y_true = np.concatenate(all_targets)
    y_pred = probs_arr.argmax(axis=1)
    avg_loss = loss_sum / max(1, n_samples)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=list(range(len(QUALITY_CLASSES))), average="macro", zero_division=0)
    )
    return avg_loss, acc, macro_f1, y_true, y_pred, probs_arr


def _fit_single_dl_run(
    model_builder,
    train_images: list[Image.Image],
    y_train: np.ndarray,
    val_images: list[Image.Image],
    y_val: np.ndarray,
    *,
    image_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    freeze_backbone_eval: bool,
    seed: int,
) -> dict[str, Any]:
    """Ejecuta un entrenamiento completo con registro por época y early stopping."""
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PreloadedFruitDataset(train_images, y_train, build_transforms(image_size, training=True))
    train_eval_ds = PreloadedFruitDataset(train_images, y_train, build_transforms(image_size, training=False))
    val_ds = PreloadedFruitDataset(val_images, y_val, build_transforms(image_size, training=False))

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_builder().to(device)

    # Pesos balanceados por frecuencia inversa: w_c = N / (K * N_c)
    counts = np.bincount(y_train, minlength=len(QUALITY_CLASSES)).astype(np.float32)
    class_weights = len(y_train) / (len(QUALITY_CLASSES) * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbone_eval and hasattr(model, "features"):
            # Mantener congeladas las estadísticas running_mean/running_var de BatchNorm
            model.features.eval()

        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.size(0))
            seen += int(labels.size(0))

        train_loss_aug = running_loss / max(1, seen)
        tr_loss, tr_acc, tr_f1, _, _, _ = _evaluate_loader(model, train_eval_loader, criterion, device)
        va_loss, va_acc, va_f1, _, _, _ = _evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(va_f1)

        history.append(
            {
                "epoch": epoch,
                "train_loss_aug": round(train_loss_aug, 4),
                "train_loss": round(tr_loss, 4),
                "val_loss": round(va_loss, 4),
                "train_acc": round(tr_acc, 4),
                "val_acc": round(va_acc, 4),
                "train_macro_f1": round(tr_f1, 4),
                "val_macro_f1": round(va_f1, 4),
            }
        )

        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    _, _, final_tr_f1, _, _, _ = _evaluate_loader(model, train_eval_loader, criterion, device)
    _, _, final_va_f1, _, _, _ = _evaluate_loader(model, val_loader, criterion, device)

    return {
        "model": model,
        "device": device,
        "image_size": image_size,
        "best_epoch": best_epoch,
        "best_params": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs_ran": len(history),
        },
        "train_macro_f1": float(final_tr_f1),
        "val_macro_f1": float(final_va_f1),
        "train_time_sec": round(elapsed, 3),
        "params_info": count_parameters(model),
        "history": history,
    }


def train_deep_learning_suite(
    train_images: list[Image.Image],
    y_train: np.ndarray,
    val_images: list[Image.Image],
    y_val: np.ndarray,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Entrena y ajusta CNN desde cero y MobileNetV2 congelada sobre las mismas particiones."""
    torch.set_num_threads(min(8, max(2, torch.get_num_threads())))
    results: dict[str, dict[str, Any]] = {}

    # 1. CNN simple desde cero (ajuste de tasa de aprendizaje sobre validación)
    cnn_trials: list[dict[str, Any]] = []
    for lr in (1e-3, 3e-4):
        run = _fit_single_dl_run(
            lambda: SimpleFruitCNN(len(QUALITY_CLASSES)),
            train_images,
            y_train,
            val_images,
            y_val,
            image_size=SCRATCH_CNN_SIZE,
            learning_rate=lr,
            weight_decay=1e-4,
            epochs=22,
            batch_size=32,
            patience=6,
            freeze_backbone_eval=False,
            seed=seed,
        )
        cnn_trials.append(run)
        print(
            f"  [CNN desde cero lr={lr}] best_ep={run['best_epoch']} "
            f"train_f1={run['train_macro_f1']:.4f} val_f1={run['val_macro_f1']:.4f} "
            f"({run['train_time_sec']:.1f}s)",
            flush=True,
        )
    best_cnn = max(cnn_trials, key=lambda r: r["val_macro_f1"])
    best_cnn["family"] = "deep_learning"
    best_cnn["arch_name"] = "simple_cnn"
    best_cnn["tuning_trials"] = [
        {"lr": r["best_params"]["learning_rate"], "val_macro_f1": r["val_macro_f1"], "best_epoch": r["best_epoch"]}
        for r in cnn_trials
    ]
    results["CNN simple (desde cero)"] = best_cnn

    # 2. Transfer Learning: MobileNetV2 congelada + cabeza densa propia
    mob_trials: list[dict[str, Any]] = []
    for lr, wd in ((1e-3, 1e-4), (3e-4, 5e-4)):
        run = _fit_single_dl_run(
            lambda: build_transfer_mobilenet_v2(len(QUALITY_CLASSES), pretrained=True),
            train_images,
            y_train,
            val_images,
            y_val,
            image_size=DL_IMAGE_SIZE,
            learning_rate=lr,
            weight_decay=wd,
            epochs=14,
            batch_size=32,
            patience=5,
            freeze_backbone_eval=True,
            seed=seed,
        )
        mob_trials.append(run)
        print(
            f"  [MobileNetV2 lr={lr}, wd={wd}] best_ep={run['best_epoch']} "
            f"train_f1={run['train_macro_f1']:.4f} val_f1={run['val_macro_f1']:.4f} "
            f"({run['train_time_sec']:.1f}s)",
            flush=True,
        )
    best_mob = max(mob_trials, key=lambda r: r["val_macro_f1"])
    best_mob["family"] = "deep_learning"
    best_mob["arch_name"] = "mobilenet_v2"
    best_mob["tuning_trials"] = [
        {
            "lr": r["best_params"]["learning_rate"],
            "weight_decay": r["best_params"]["weight_decay"],
            "val_macro_f1": r["val_macro_f1"],
            "best_epoch": r["best_epoch"],
        }
        for r in mob_trials
    ]
    results["MobileNetV2 (Transfer)"] = best_mob

    return results


def predict_dl_dataset(
    model: nn.Module,
    images_pil: list[Image.Image],
    *,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(y_pred, probs)`` sobre una lista de imágenes PIL."""
    dummy_y = np.zeros(len(images_pil), dtype=np.int64)
    ds = PreloadedFruitDataset(images_pil, dummy_y, build_transforms(image_size, training=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    probs_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            probs_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs_batches)
    return probs.argmax(axis=1), probs


def predict_single_pil(
    model: nn.Module,
    image_pil: Image.Image,
    *,
    image_size: int,
    device: torch.device,
    use_foreground_crop: bool = True,
) -> tuple[str, dict[str, float]]:
    """Infiere calidad y probabilidades para una sola imagen PIL (uso en app Streamlit)."""
    working = image_pil.convert("RGB")
    if use_foreground_crop:
        cropped_rgb, _ = crop_fruit_foreground(np.asarray(working))
        working = Image.fromarray(cropped_rgb)
    tf = build_transforms(image_size, training=False)
    tensor = tf(working).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    prob_map = {cls: float(probs[i]) for i, cls in enumerate(QUALITY_CLASSES)}
    return QUALITY_CLASSES[pred_idx], prob_map


def save_dl_checkpoint(
    model: nn.Module,
    arch_name: str,
    image_size: int,
    path: Path,
) -> None:
    """Guarda pesos y arquitectura de un modelo PyTorch para despliegue offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch_name": arch_name,
        "image_size": int(image_size),
        "classes": list(QUALITY_CLASSES),
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(payload, path)


def load_dl_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any], torch.device]:
    """Carga un checkpoint PyTorch guardado sin requerir conexión a internet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=True)
    arch = payload["arch_name"]
    num_classes = len(payload["classes"])
    if arch == "mobilenet_v2":
        model = build_transfer_mobilenet_v2(num_classes, pretrained=False)
    elif arch == "simple_cnn":
        model = SimpleFruitCNN(num_classes)
    else:
        raise ValueError(f"Arquitectura desconocida en checkpoint: {arch}")
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    meta = {
        "arch_name": arch,
        "image_size": int(payload["image_size"]),
        "classes": list(payload["classes"]),
    }
    return model, meta, device
