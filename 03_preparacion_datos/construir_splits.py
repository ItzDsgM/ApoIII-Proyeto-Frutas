"""Construye el dataset combinado reproducible y sus particiones sin fuga (train/val/test).

Combina las tres fuentes exigidas/permitidas por el enunciado (secciones 2.1–2.3):
1. Fotos propias validadas por el equipo (``data/annotations/own_annotations.csv``).
2. Frutas individuales segmentadas de ``Mixed Qualit_Fruits`` y validadas manualmente
   (``data/annotations/mixed_crops_annotations.csv``).
3. Muestra estratificada y equiespaciada de Kaggle ``Good Quality_Fruits`` y
   ``Bad Quality_Fruits`` (``data/external/``) para balancear las clases y aportar
   diversidad entre especies y fondos.

Prevención de fuga de datos (data leakage):
- Todos los recortes provenientes de una misma foto de ``Mixed Qualit_Fruits`` comparten
  ``group_id`` (``source_group_id``).
- Además, se calcula un hash perceptual ``dHash`` de 64 bits sobre cada imagen y se
  agrupan imágenes casi idénticas (distancia de Hamming <= 6 dentro de la misma especie
  y fuente) mediante Union-Find antes de ejecutar ``StratifiedGroupKFold``.

Genera:
- ``data/annotations/fruitnet_splits.csv``: tabla maestra con partición ``train`` / ``val`` / ``test``.
- ``data/annotations/size_references.csv``: referencias geométricas históricas alineadas
  al diámetro normalizado invariante a orientación ``max(bw, bh) / max(W, H)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNOTATIONS_DIR,
    EXTERNAL_DATA_DIR,
    FRUITNET_SPLITS_CSV,
    IMAGE_EXTENSIONS,
    KAGGLE_QUALITY_BUCKETS,
    KAGGLE_SPECIES_MAP,
    MIXED_ANNOTATIONS_CSV,
    OWN_ANNOTATIONS_CSV,
    QUALITY_CLASSES,
    RANDOM_SEED,
)
from src.preprocessing import load_clean_rgb, sha256_file

SIZE_REFERENCES_CSV = ANNOTATIONS_DIR / "size_references.csv"


def compute_dhash(image_path: Path, hash_size: int = 8) -> np.ndarray:
    """Calcula un hash perceptual por diferencias horizontales (64 bits)."""
    with Image.open(image_path) as raw:
        gray = raw.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()


def assign_perceptual_groups(
    frame: pd.DataFrame,
    hamming_threshold: int = 6,
) -> pd.DataFrame:
    """Agrupa duplicados exactos, recortes de misma escena y vistas casi idénticas."""
    df = frame.reset_index(drop=True).copy()
    n = len(df)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 1. Agrupar por sha256 exacto y por group_id de origen (ej. misma foto Mixed)
    for col in ("sha256", "group_id"):
        seen: dict[str, int] = {}
        for idx, val in enumerate(df[col].fillna("")):
            key = str(val).strip()
            if not key:
                continue
            if key in seen:
                union(idx, seen[key])
            else:
                seen[key] = idx

    # 2. Agrupar imágenes perceptualmente casi idénticas (misma especie y fuente)
    dhashes = [compute_dhash(PROJECT_ROOT / p) for p in df["image_path"]]
    buckets: dict[tuple[str, str], list[int]] = {}
    for idx, row in df.iterrows():
        buckets.setdefault((str(row["source"]), str(row["species"])), []).append(idx)

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        mat = np.vstack([dhashes[i] for i in indices])
        for pos_i in range(len(indices)):
            # Distancia de Hamming vectorizada contra los siguientes elementos del bloque
            diffs = np.count_nonzero(mat[pos_i + 1 :] != mat[pos_i], axis=1)
            matches = np.flatnonzero(diffs <= hamming_threshold)
            for offset in matches:
                union(indices[pos_i], indices[pos_i + 1 + int(offset)])

    df["leakage_group"] = [f"grp_{find(i):04d}" for i in range(n)]
    return df


def sample_kaggle_good_bad(
    external_dir: Path,
    good_per_species: int = 45,
    bad_per_species: int = 100,
) -> list[dict[str, object]]:
    """Muestrea equiespaciadamente imágenes de ``Good Quality_Fruits`` y ``Bad Quality_Fruits``."""
    rows: list[dict[str, object]] = []
    for bucket_folder, quality_label in KAGGLE_QUALITY_BUCKETS.items():
        bucket_dir = external_dir / bucket_folder
        if not bucket_dir.is_dir():
            matches = [p for p in external_dir.rglob(bucket_folder) if p.is_dir()]
            if not matches:
                raise FileNotFoundError(f"No se encontró {bucket_folder} dentro de {external_dir}")
            bucket_dir = matches[0]

        limit = good_per_species if quality_label == "Buena" else bad_per_species
        for sp_dir in sorted(p for p in bucket_dir.iterdir() if p.is_dir()):
            raw_prefix = sp_dir.name.split("_")[0].strip().lower()
            species = KAGGLE_SPECIES_MAP.get(raw_prefix, raw_prefix)

            files = sorted(
                p for p in sp_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                continue
            n_pick = min(limit, len(files))
            indices = np.linspace(0, len(files) - 1, num=n_pick, dtype=int)
            selected = [files[int(i)] for i in sorted(set(indices.tolist()))]

            for img_path in selected:
                img = load_clean_rgb(img_path)
                w, h = img.size
                digest = sha256_file(img_path)
                rows.append(
                    {
                        "image_path": img_path.relative_to(PROJECT_ROOT).as_posix(),
                        "species": species,
                        "quality": quality_label,
                        "width": w,
                        "height": h,
                        "sha256": digest,
                        "source": "FruitNet_Kaggle",
                        "group_id": f"kaggle_{species}_{digest[:12]}",
                    }
                )
    return rows


def build_combined_splits(
    good_per_species: int = 45,
    bad_per_species: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Combina todas las fuentes, asigna grupos anti-fuga y divide en train/val/test."""
    own_df = pd.read_csv(OWN_ANNOTATIONS_CSV)
    own_sub = own_df[["image_path", "species", "quality", "width", "height", "sha256", "source", "group_id"]].copy()

    mixed_frames = []
    if MIXED_ANNOTATIONS_CSV.is_file() and MIXED_ANNOTATIONS_CSV.stat().st_size > 0:
        mixed_df = pd.read_csv(MIXED_ANNOTATIONS_CSV)
        mixed_frames.append(
            mixed_df[["image_path", "species", "quality", "width", "height", "sha256", "source", "group_id"]].copy()
        )

    kaggle_rows = sample_kaggle_good_bad(
        EXTERNAL_DATA_DIR,
        good_per_species=good_per_species,
        bad_per_species=bad_per_species,
    )
    kaggle_df = pd.DataFrame(kaggle_rows)

    combined = pd.concat([own_sub, *mixed_frames, kaggle_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
    combined = assign_perceptual_groups(combined, hamming_threshold=6)

    # Estratificación combinada calidad + origen simplificado para garantizar fotos propias y Kaggle en todos los splits
    strat_target = combined["quality"].astype(str) + "__" + np.where(
        combined["source"].eq("Propia"), "Propia", "Publica"
    )
    # Si algún estrato compuesto tiene menos de 5 grupos, caer en estratificación por calidad
    group_strat_counts = combined.groupby(strat_target)["leakage_group"].nunique()
    if int(group_strat_counts.min()) < 5:
        strat_target = combined["quality"].astype(str)

    # 1. Separar 20 % para conjunto de prueba reservado (test)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(combined, strat_target, combined["leakage_group"]))

    # 2. Separar ~15 % del total (~19 % de dev) para validación (val)
    dev_frame = combined.iloc[dev_idx]
    dev_strat = strat_target.iloc[dev_idx]
    if int(dev_frame.groupby(dev_strat)["leakage_group"].nunique().min()) < 5:
        dev_strat = dev_frame["quality"].astype(str)

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    tr_rel_idx, val_rel_idx = next(inner.split(dev_frame, dev_strat, dev_frame["leakage_group"]))
    train_idx = dev_idx[tr_rel_idx]
    val_idx = dev_idx[val_rel_idx]

    splits = np.full(len(combined), "", dtype=object)
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    combined["split"] = splits

    # Verificación estricta de cero fuga de grupos entre particiones
    tr_g = set(combined.loc[combined["split"].eq("train"), "leakage_group"])
    va_g = set(combined.loc[combined["split"].eq("val"), "leakage_group"])
    te_g = set(combined.loc[combined["split"].eq("test"), "leakage_group"])
    if (tr_g & va_g) or (tr_g & te_g) or (va_g & te_g):
        raise RuntimeError("Se detectó fuga de grupos entre train, val y test.")

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FRUITNET_SPLITS_CSV, index=False)
    return combined


def align_historical_size_references(old_data_dir: Path | None) -> int:
    """Alinea las 204 cajas de referencia históricas al CSV limpio ``size_references.csv``."""
    if old_data_dir is None or not old_data_dir.is_dir():
        return 0

    ref_by_sha: dict[str, dict[str, object]] = {}
    for fname in (
        "own_size_visual_reference.json",
        "own_size_visual_reference_supplement_2026-11-16.json",
        "own_size_completion_limon_tomate_2026-11-16.json",
        "own_size_completion_mango_naranja_2026-11-16.json",
    ):
        fpath = old_data_dir / fname
        if not fpath.is_file():
            continue
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        rows = payload.get("rows") or payload.get("added_rows") or payload.get("references") or []
        for row in rows:
            ratio_old = row.get("reference_ratio")
            orig_size = row.get("original_size")
            if ratio_old is None or not orig_size:
                continue
            ow, oh = float(orig_size[0]), float(orig_size[1])
            max_extent_px = float(ratio_old) * ow
            # Convertir de max(bw,bh)/W a la métrica invariante a rotación max(bw,bh)/max(W,H)
            norm_diam = max_extent_px / max(ow, oh)
            ref_by_sha[row["sha256"]] = {
                "reference_normalized_diameter": round(norm_diam, 6),
                "reference_orig_width": int(ow),
                "reference_orig_height": int(oh),
            }

    own_df = pd.read_csv(OWN_ANNOTATIONS_CSV)
    aligned_rows: list[dict[str, object]] = []
    for _, row in own_df.iterrows():
        orig_sha = str(row["original_sha256"])
        if orig_sha in ref_by_sha:
            aligned_rows.append(
                {
                    "image_path": row["image_path"],
                    "species": row["species"],
                    "quality": row["quality"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "reference_normalized_diameter": ref_by_sha[orig_sha]["reference_normalized_diameter"],
                    "reference_method": "visual_bbox_max_extent_over_max_side",
                }
            )

    if aligned_rows:
        pd.DataFrame(aligned_rows).to_csv(SIZE_REFERENCES_CSV, index=False)
    return len(aligned_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good-per-species", type=int, default=45)
    parser.add_argument("--bad-per-species", type=int, default=100)
    parser.add_argument(
        "--old-size-refs-dir",
        type=Path,
        default=None,
        help="Carpeta data/ del repositorio previo para migrar las 204 referencias geométricas.",
    )
    args = parser.parse_args()

    combined = build_combined_splits(
        good_per_species=args.good_per_species,
        bad_per_species=args.bad_per_species,
    )
    print(f"Dataset combinado guardado en {FRUITNET_SPLITS_CSV.relative_to(PROJECT_ROOT)}: {len(combined)} imágenes")
    print(f"Grupos perceptuales/fuente únicos: {combined['leakage_group'].nunique()}")
    print("\nDistribución por split y clase:")
    print(pd.crosstab(combined["split"], combined["quality"], margins=True))
    print("\nDistribución por origen y clase:")
    print(pd.crosstab(combined["source"], combined["quality"], margins=True))
    print("\nDistribución por especie y clase:")
    print(pd.crosstab(combined["species"], combined["quality"], margins=True))

    if args.old_size_refs_dir is not None:
        n_refs = align_historical_size_references(args.old_size_refs_dir)
        print(f"\nReferencias geométricas alineadas en {SIZE_REFERENCES_CSV.relative_to(PROJECT_ROOT)}: {n_refs}")


if __name__ == "__main__":
    main()
