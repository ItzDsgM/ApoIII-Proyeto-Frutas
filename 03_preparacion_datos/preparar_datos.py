"""Prepara, limpia, renombra y consolida las fotos propias y los recortes de Mixed.

Operaciones ejecutadas (Fase 3 CRISP-DM - Preparación de datos):
1. Recorre dinámicamente ``data/own_photos/<especie>/<Buena|Regular|Mala>/``
   (tolera carpetas vacías o inexistentes).
2. Detecta y excluye duplicados binarios exactos por SHA-256 (registrándolos en
   ``data/annotations/excluded_own_photos.csv``).
3. Aplica orientación EXIF, elimina todos los metadatos EXIF/GPS (privacidad - PI1)
   y redimensiona a lado mayor máximo ``1024 px`` con interpolación Lanczos.
4. Renombra cada imagen admitida con el formato simple acordado:
       ``{especie}_{nnn:03d}_{calidad_minuscula}.jpg``
   (ej.: ``mango_001_buena.jpg``, ``limon_012_regular.jpg``).
5. Genera el CSV maestro ``data/annotations/own_annotations.csv``.
6. Si existen recortes clasificados manualmente en
   ``data/mixed_staging/<especie>/<Buena|Regular|Mala>/``, los consolida en
   ``data/mixed_crops/<especie>/<calidad>/`` y genera
   ``data/annotations/mixed_crops_annotations.csv`` conservando ``source_group_id``
   para evitar fuga de datos entre particiones.

Uso recomendado:
    # Primero simular sin modificar ningún archivo:
    .venv/bin/python 03_preparacion_datos/preparar_datos.py --dry-run

    # Cuando estés conforme con el resumen, aplicar los cambios:
    .venv/bin/python 03_preparacion_datos/preparar_datos.py --apply --reviewer "Equipo"
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNOTATIONS_DIR,
    DATA_DIR,
    EXCLUDED_OWN_CSV,
    IMAGE_EXTENSIONS,
    MAX_WORKING_SIDE_PX,
    MIXED_ANNOTATIONS_CSV,
    MIXED_STAGING_DIR,
    OWN_ANNOTATIONS_CSV,
    OWN_PHOTOS_DIR,
    QUALITY_CLASSES,
)
from src.preprocessing import load_clean_rgb, save_clean_jpeg, sha256_file
from src.size_estimation import classify_size_category

MIXED_CROPS_DIR = DATA_DIR / "mixed_crops"
SIZE_REFERENCES_CSV = ANNOTATIONS_DIR / "size_references.csv"

OWN_CSV_FIELDS = [
    "image_path",
    "species",
    "quality",
    "width",
    "height",
    "file_bytes",
    "sha256",
    "original_filename",
    "original_sha256",
    "source",
    "group_id",
    "reviewer",
    "reference_normalized_diameter",
    "size_category",
]

EXCLUDED_CSV_FIELDS = [
    "original_path",
    "species",
    "original_folder_label",
    "original_sha256",
    "reason",
]

MIXED_CSV_FIELDS = [
    "image_path",
    "species",
    "quality",
    "width",
    "height",
    "sha256",
    "source",
    "source_relative_path",
    "group_id",
    "reviewer",
]


def _normalize_label(folder_name: str) -> str | None:
    """Normaliza el nombre de subcarpeta a una clase oficial de ``QUALITY_CLASSES``."""
    lookup = {cls.casefold(): cls for cls in QUALITY_CLASSES}
    return lookup.get(folder_name.strip().casefold())


def _load_size_lookup() -> dict[str, tuple[str, str]]:
    """Carga anotaciones de tamaño previas por ``image_path`` para preservarlas al regenerar CSV."""
    size_lookup: dict[str, tuple[str, str]] = {}
    if OWN_ANNOTATIONS_CSV.is_file():
        with OWN_ANNOTATIONS_CSV.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                diam = (row.get("reference_normalized_diameter") or "").strip()
                cat = (row.get("size_category") or "").strip()
                if diam:
                    cat = cat or classify_size_category(float(diam))
                    size_lookup[row["image_path"]] = (diam, cat)
    if SIZE_REFERENCES_CSV.is_file():
        with SIZE_REFERENCES_CSV.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                diam = (row.get("reference_normalized_diameter") or "").strip()
                cat = (row.get("reference_size_category") or "").strip()
                if diam:
                    cat = cat or classify_size_category(float(diam))
                    size_lookup[row["image_path"]] = (diam, cat)
    return size_lookup


def process_own_photos(
    own_dir: Path = OWN_PHOTOS_DIR,
    *,
    apply: bool = False,
    max_side: int = MAX_WORKING_SIDE_PX,
    reviewer: str = "Equipo",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Inspecciona o procesa todas las fotos propias en ``data/own_photos/``."""
    if not own_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de fotos propias: {own_dir}")

    size_lookup = _load_size_lookup()
    seen_hashes: dict[str, str] = {}
    admitted_plan: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []

    for species_dir in sorted(p for p in own_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        species = species_dir.name.strip().lower()
        species_counter = 0

        for cls in QUALITY_CLASSES:
            cls_dir = next(
                (p for p in species_dir.iterdir() if p.is_dir() and _normalize_label(p.name) == cls),
                None,
            )
            if cls_dir is None:
                continue

            image_files = sorted(
                p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for img_path in image_files:
                orig_sha = sha256_file(img_path)
                rel_orig = img_path.relative_to(PROJECT_ROOT).as_posix()

                if orig_sha in seen_hashes:
                    excluded_rows.append(
                        {
                            "original_path": rel_orig,
                            "species": species,
                            "original_folder_label": cls,
                            "original_sha256": orig_sha,
                            "reason": f"Duplicado binario exacto de {seen_hashes[orig_sha]}",
                        }
                    )
                    if apply:
                        img_path.unlink()
                    continue

                seen_hashes[orig_sha] = rel_orig
                species_counter += 1
                new_name = f"{species}_{species_counter:03d}_{cls.lower()}.jpg"
                target_path = cls_dir / new_name
                admitted_plan.append(
                    {
                        "src_path": img_path,
                        "target_path": target_path,
                        "species": species,
                        "quality": cls,
                        "original_filename": img_path.name,
                        "original_sha256": orig_sha,
                    }
                )

        # Registrar cualquier imagen que el usuario haya dejado en Descartar/ si existe
        discard_dir = next(
            (p for p in species_dir.iterdir() if p.is_dir() and p.name.casefold() == "descartar"),
            None,
        )
        if discard_dir is not None:
            for img_path in sorted(
                p for p in discard_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ):
                excluded_rows.append(
                    {
                        "original_path": img_path.relative_to(PROJECT_ROOT).as_posix(),
                        "species": species,
                        "original_folder_label": "Descartar",
                        "original_sha256": sha256_file(img_path),
                        "reason": "Descartada manualmente en revisión del equipo",
                    }
                )

    own_rows: list[dict[str, object]] = []
    if not apply:
        for item in admitted_plan:
            img = load_clean_rgb(item["src_path"], max_side=max_side)
            w, h = img.size
            rel_target = item["target_path"].relative_to(PROJECT_ROOT).as_posix()
            ref_diam, ref_cat = size_lookup.get(rel_target, ("", ""))
            own_rows.append(
                {
                    "image_path": rel_target,
                    "species": item["species"],
                    "quality": item["quality"],
                    "width": w,
                    "height": h,
                    "file_bytes": item["src_path"].stat().st_size,
                    "sha256": "(dry-run)",
                    "original_filename": item["original_filename"],
                    "original_sha256": item["original_sha256"],
                    "source": "Propia",
                    "group_id": f"own_{item['species']}_{item['original_sha256'][:12]}",
                    "reviewer": reviewer,
                    "reference_normalized_diameter": ref_diam,
                    "size_category": ref_cat,
                }
            )
        return own_rows, excluded_rows

    # Escritura segura en dos fases (carpeta temporal -> destino final) para evitar colisiones
    with tempfile.TemporaryDirectory(prefix="prep_own_") as tmp_name:
        tmp_dir = Path(tmp_name)
        staged: list[tuple[Path, dict[str, object]]] = []

        for idx, item in enumerate(admitted_plan):
            clean_img = load_clean_rgb(item["src_path"], max_side=max_side)
            tmp_file = tmp_dir / f"{idx:05d}.jpg"
            w, h = save_clean_jpeg(clean_img, tmp_file, max_side=max_side)
            item["width"] = w
            item["height"] = h
            staged.append((tmp_file, item))

        # Eliminar los archivos fuente originales ya convertidos antes de mover los definitivos
        for item in admitted_plan:
            src_path: Path = item["src_path"]
            if src_path.exists():
                src_path.unlink()

        for tmp_file, item in staged:
            target_path: Path = item["target_path"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_file), str(target_path))
            final_sha = sha256_file(target_path)
            rel_target = target_path.relative_to(PROJECT_ROOT).as_posix()
            ref_diam, ref_cat = size_lookup.get(rel_target, ("", ""))
            own_rows.append(
                {
                    "image_path": rel_target,
                    "species": item["species"],
                    "quality": item["quality"],
                    "width": item["width"],
                    "height": item["height"],
                    "file_bytes": target_path.stat().st_size,
                    "sha256": final_sha,
                    "original_filename": item["original_filename"],
                    "original_sha256": item["original_sha256"],
                    "source": "Propia",
                    "group_id": f"own_{item['species']}_{final_sha[:12]}",
                    "reviewer": reviewer,
                    "reference_normalized_diameter": ref_diam,
                    "size_category": ref_cat,
                }
            )

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with OWN_ANNOTATIONS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OWN_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(own_rows)

    with EXCLUDED_OWN_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXCLUDED_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(excluded_rows)

    return own_rows, excluded_rows


def process_mixed_crops(
    staging_dir: Path = MIXED_STAGING_DIR,
    output_dir: Path = MIXED_CROPS_DIR,
    *,
    apply: bool = False,
    reviewer: str = "Equipo",
) -> list[dict[str, object]]:
    """Consolida recortes de ``Mixed Qualit_Fruits`` ya clasificados por el equipo."""
    manifest_path = staging_dir / "mixed_crops_manifest.csv"
    manifest_lookup: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                manifest_lookup[row["crop_filename"]] = row

    if not staging_dir.is_dir():
        return []

    rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()

    for species_dir in sorted(p for p in staging_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        species = species_dir.name.strip().lower()
        species_counter = 0

        for cls in QUALITY_CLASSES:
            cls_dir = next(
                (p for p in species_dir.iterdir() if p.is_dir() and _normalize_label(p.name) == cls),
                None,
            )
            if cls_dir is None:
                continue

            image_files = sorted(
                p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for img_path in image_files:
                digest = sha256_file(img_path)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                species_counter += 1

                meta = manifest_lookup.get(img_path.name, {})
                new_name = f"mixed_{species}_{species_counter:03d}_{cls.lower()}.jpg"
                dst_path = output_dir / species / cls / new_name
                clean_img = load_clean_rgb(img_path)

                if apply:
                    w, h = save_clean_jpeg(clean_img, dst_path, max_side=None, quality=92)
                    final_sha = sha256_file(dst_path)
                else:
                    w, h = clean_img.size
                    final_sha = "(dry-run)"

                rows.append(
                    {
                        "image_path": dst_path.relative_to(PROJECT_ROOT).as_posix(),
                        "species": species,
                        "quality": cls,
                        "width": w,
                        "height": h,
                        "sha256": final_sha,
                        "source": "FruitNet_Mixed_Segmentada",
                        "source_relative_path": meta.get("source_relative_path", ""),
                        "group_id": meta.get("source_group_id", f"mixed_{species}_{digest[:12]}"),
                        "reviewer": reviewer,
                    }
                )

    if apply and rows:
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        with MIXED_ANNOTATIONS_CSV.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MIXED_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def print_summary(
    own_rows: list[dict[str, object]],
    excluded_rows: list[dict[str, object]],
    mixed_rows: list[dict[str, object]],
    *,
    apply: bool,
) -> None:
    """Muestra un resumen claro por especie y clase."""
    mode = "APLICADO" if apply else "SIMULACIÓN (--dry-run, sin modificar archivos)"
    print(f"\n=== Preparación de datos [{mode}] ===")
    print(f"Fotos propias válidas: {len(own_rows)} | Excluidas/duplicadas: {len(excluded_rows)}")

    by_sp_cls = Counter((r["species"], r["quality"]) for r in own_rows)
    by_cls = Counter(r["quality"] for r in own_rows)
    species_all = sorted({r["species"] for r in own_rows})
    for sp in species_all:
        counts = ", ".join(f"{cls}: {by_sp_cls[(sp, cls)]}" for cls in QUALITY_CLASSES)
        print(f"  - {sp:14s} -> {counts}")
    print("  Total propias por clase:", dict(by_cls))

    if apply and own_rows:
        total_mb = sum(int(r["file_bytes"]) for r in own_rows) / 1e6
        print(f"  Peso total optimizado (sin EXIF, <=1024px): {total_mb:.2f} MB")
        print(f"  CSV generado: {OWN_ANNOTATIONS_CSV.relative_to(PROJECT_ROOT)}")

    if mixed_rows:
        m_cls = Counter(r["quality"] for r in mixed_rows)
        m_sp = Counter((r["species"], r["quality"]) for r in mixed_rows)
        print(f"\nRecortes clasificados de Mixed Qualit_Fruits: {len(mixed_rows)}")
        for sp in sorted({r["species"] for r in mixed_rows}):
            counts = ", ".join(f"{cls}: {m_sp[(sp, cls)]}" for cls in QUALITY_CLASSES)
            print(f"  - {sp:14s} -> {counts}")
        print("  Total Mixed por clase:", dict(m_cls))
    else:
        print(
            "\nRecortes de Mixed clasificados: 0 "
            "(clasifica primero en data/mixed_staging/<especie>/<Buena|Regular|Mala>/)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Simula la limpieza y muestra el resumen sin modificar archivos (por defecto).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Aplica deduplicación, limpieza EXIF, redimensionado, renombrado y guarda los CSV.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=MAX_WORKING_SIDE_PX,
        help=f"Lado máximo en píxeles para las fotos propias (por defecto: {MAX_WORKING_SIDE_PX}).",
    )
    parser.add_argument(
        "--reviewer",
        type=str,
        default="Equipo",
        help="Identificador de quien validó las etiquetas (por defecto: Equipo).",
    )
    args = parser.parse_args()

    own_rows, excluded_rows = process_own_photos(
        apply=args.apply, max_side=args.max_side, reviewer=args.reviewer
    )
    mixed_rows = process_mixed_crops(apply=args.apply, reviewer=args.reviewer)
    print_summary(own_rows, excluded_rows, mixed_rows, apply=args.apply)


if __name__ == "__main__":
    main()
