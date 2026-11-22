"""Segmenta frutas individuales desde ``Mixed Qualit_Fruits`` (Enunciado sec. 2.1).

Extrae ejemplares individuales bien separados de imágenes multifruto de Kaggle
usando distancia cromática en CIELAB + Otsu + Watershed, y los guarda en
``data/mixed_staging/<especie>/Por_revisar/`` junto con las carpetas destino
``Buena/``, ``Regular/``, ``Mala/`` y ``Descartar/`` para clasificación humana.

Uso desde la raíz del proyecto:
    .venv/bin/python 03_preparacion_datos/segmentar_mixed.py --max-sources 25
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXTERNAL_DATA_DIR,
    IMAGE_EXTENSIONS,
    KAGGLE_MIXED_FOLDERS,
    KAGGLE_SPECIES_MAP,
    MIXED_STAGING_DIR,
    QUALITY_CLASSES,
)
from src.preprocessing import load_clean_rgb, save_clean_jpeg, segment_multi_fruit_instances


def find_mixed_root(external_dir: Path) -> Path:
    """Localiza la carpeta ``Mixed Qualit_Fruits`` dentro de ``data/external``."""
    for folder_name in KAGGLE_MIXED_FOLDERS:
        direct = external_dir / folder_name
        if direct.is_dir():
            return direct
        matches = [p for p in external_dir.rglob(folder_name) if p.is_dir()]
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No se encontró ninguna carpeta {KAGGLE_MIXED_FOLDERS} dentro de {external_dir}."
    )


def select_equispaced_files(files: list[Path], limit: int) -> list[Path]:
    """Selecciona ``limit`` archivos equiespaciados para evitar fotogramas consecutivos repetidos."""
    if limit <= 0 or len(files) <= limit:
        return files
    indices = np.linspace(0, len(files) - 1, num=limit, dtype=int)
    return [files[int(i)] for i in sorted(set(indices.tolist()))]


def save_preview_grid(
    previews: list[tuple[str, np.ndarray]],
    output_path: Path,
    cols: int = 4,
    tile_w: int = 320,
    tile_h: int = 240,
) -> None:
    """Genera una hoja de contacto para inspeccionar visualmente las cajas detectadas."""
    if not previews:
        return
    rows = (len(previews) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, img_rgb) in enumerate(previews):
        r, c = divmod(idx, cols)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((tile_w - 8, tile_h - 24), Image.Resampling.LANCZOS)
        x0 = c * tile_w + (tile_w - pil_img.width) // 2
        y0 = r * tile_h + 20 + (tile_h - 24 - pil_img.height) // 2
        canvas.paste(pil_img, (x0, y0))
        draw.text((c * tile_w + 6, r * tile_h + 4), title[:42], fill=(20, 20, 20))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=90)


def is_grass_background(image_rgb: np.ndarray) -> bool:
    """Detecta si el borde de la escena corresponde predominantemente a césped."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h = hsv.shape[0]
    strip = max(8, h // 15)
    border = np.concatenate([hsv[:strip, :].reshape(-1, 3), hsv[-strip:, :].reshape(-1, 3)], axis=0)
    green_mask = (border[:, 0] >= 35) & (border[:, 0] <= 85) & (border[:, 1] >= 50)
    return float(np.mean(green_mask)) > 0.35


def is_grass_patch(crop_rgb: np.ndarray) -> bool:
    """Filtra falsos positivos de vegetación/césped (alta proporción verde + alta textura)."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    green_frac = float(np.mean((hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] >= 45)))
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    edge_energy = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return green_frac > 0.55 and edge_energy > 650.0


def segment_mixed_dataset(
    external_dir: Path = EXTERNAL_DATA_DIR,
    staging_dir: Path = MIXED_STAGING_DIR,
    max_sources_per_species: int = 45,
    max_crops_per_species: int = 80,
) -> dict[str, int]:
    """Segmenta frutas individuales de ``Mixed Qualit_Fruits`` por especie."""
    mixed_root = find_mixed_root(external_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    preview_tiles: list[tuple[str, np.ndarray]] = []
    counts_by_species: dict[str, int] = {}

    for species_dir in sorted(p for p in mixed_root.iterdir() if p.is_dir()):
        raw_name = species_dir.name.strip().lower()
        species = KAGGLE_SPECIES_MAP.get(raw_name, raw_name)

        # Crear estructura de revisión manual para esta especie
        review_dir = staging_dir / species / "Por_revisar"
        review_dir.mkdir(parents=True, exist_ok=True)
        for cls in (*QUALITY_CLASSES, "Descartar"):
            (staging_dir / species / cls).mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in species_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        # En banano hay >200 fotos sobre fondo uniforme; priorizarlas evita recortes de césped
        if species == "banano":
            plain_files = [p for p in image_files if not is_grass_background(np.asarray(load_clean_rgb(p)))]
            if len(plain_files) >= max_sources_per_species:
                image_files = plain_files

        selected_sources = select_equispaced_files(image_files, max_sources_per_species)

        saved_count = 0
        previews_for_species = 0
        for src_idx, img_path in enumerate(selected_sources, start=1):
            if saved_count >= max_crops_per_species:
                break
            image_pil = load_clean_rgb(img_path)
            image_rgb = np.asarray(image_pil)
            crops = [c for c in segment_multi_fruit_instances(image_rgb) if not is_grass_patch(c.crop_rgb)]
            if not crops:
                continue

            if previews_for_species < 3:
                annotated = image_rgb.copy()
                for crop_idx, crop in enumerate(crops, start=1):
                    x1, y1, x2, y2 = crop.bbox_xyxy
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 60, 0), 2)
                    cv2.putText(
                        annotated,
                        str(crop_idx),
                        (x1 + 2, min(annotated.shape[0] - 4, y1 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                preview_tiles.append((f"{species}: {img_path.name} ({len(crops)})", annotated))
                previews_for_species += 1

            rel_source = img_path.relative_to(external_dir).as_posix()
            source_group = f"mixed_{species}_{img_path.stem}"

            for crop_idx, crop in enumerate(crops, start=1):
                if saved_count >= max_crops_per_species:
                    break
                saved_count += 1
                crop_filename = f"mixed_{species}_{saved_count:03d}_src{src_idx:02d}_c{crop_idx:02d}.jpg"
                dst_path = review_dir / crop_filename
                crop_w, crop_h = save_clean_jpeg(Image.fromarray(crop.crop_rgb), dst_path, max_side=None, quality=92)
                x1, y1, x2, y2 = crop.bbox_xyxy
                manifest_rows.append(
                    {
                        "crop_filename": crop_filename,
                        "species": species,
                        "source_relative_path": rel_source,
                        "source_group_id": source_group,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "crop_width": crop_w,
                        "crop_height": crop_h,
                        "area_ratio": round(crop.area_ratio, 4),
                        "solidity": round(crop.solidity, 4),
                    }
                )

        counts_by_species[species] = saved_count

    manifest_path = staging_dir / "mixed_crops_manifest.csv"
    fieldnames = [
        "crop_filename",
        "species",
        "source_relative_path",
        "source_group_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_width",
        "crop_height",
        "area_ratio",
        "solidity",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    save_preview_grid(preview_tiles, staging_dir / "_preview_segmentacion.jpg", cols=3)
    return counts_by_species


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-sources",
        type=int,
        default=25,
        help="Máximo de imágenes multifruto equiespaciadas por especie (por defecto: 25).",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=80,
        help="Máximo de recortes individuales guardados por especie (por defecto: 80).",
    )
    args = parser.parse_args()

    counts = segment_mixed_dataset(
        max_sources_per_species=args.max_sources,
        max_crops_per_species=args.max_crops,
    )
    print("Recortes individuales generados en data/mixed_staging/:")
    total = 0
    for species, count in sorted(counts.items()):
        print(f"  - {species}: {count} recortes")
        total += count
    print(f"Total: {total} recortes. Vista previa: data/mixed_staging/_preview_segmentacion.jpg")


if __name__ == "__main__":
    main()
