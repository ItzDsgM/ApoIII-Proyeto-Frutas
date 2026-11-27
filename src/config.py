"""Configuración centralizada de rutas, clases, especies y parámetros del proyecto."""

from __future__ import annotations

from pathlib import Path

# Raíz del proyecto y carpetas principales (estructura CRISP-DM)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIR_NEGOCIO = PROJECT_ROOT / "01_comprension_negocio"
DIR_COMPRENSION_DATOS = PROJECT_ROOT / "02_comprension_datos"
DIR_PREPARACION = PROJECT_ROOT / "03_preparacion_datos"
DIR_MODELADO = PROJECT_ROOT / "04_modelado"
DIR_EVALUACION = PROJECT_ROOT / "05_evaluacion"
DIR_METRICAS = DIR_EVALUACION / "metricas"
DIR_FIGURAS = DIR_EVALUACION / "figuras"
DIR_DESPLIEGUE = PROJECT_ROOT / "06_despliegue"
DIR_MODELOS = DIR_DESPLIEGUE / "models"

# Rutas de datos
DATA_DIR = PROJECT_ROOT / "data"
OWN_PHOTOS_DIR = DATA_DIR / "own_photos"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MIXED_STAGING_DIR = DATA_DIR / "mixed_staging"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

OWN_ANNOTATIONS_CSV = ANNOTATIONS_DIR / "own_annotations.csv"
EXCLUDED_OWN_CSV = ANNOTATIONS_DIR / "excluded_own_photos.csv"
FRUITNET_SPLITS_CSV = ANNOTATIONS_DIR / "fruitnet_splits.csv"
MIXED_ANNOTATIONS_CSV = ANNOTATIONS_DIR / "mixed_crops_annotations.csv"

# Semilla global para reproducibilidad
RANDOM_SEED = 42

# Esquema oficial de 3 clases de calidad visual (en español)
QUALITY_CLASSES: tuple[str, ...] = ("Buena", "Regular", "Mala")
QUALITY_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(QUALITY_CLASSES)}
IDX_TO_QUALITY: dict[int, str] = {idx: label for idx, label in enumerate(QUALITY_CLASSES)}

QUALITY_DEFINITIONS: dict[str, str] = {
    "Buena": (
        "Fruta fresca e íntegra en la superficie visible; color normal para su "
        "especie y sin golpes ni manchas significativas."
    ),
    "Regular": (
        "Fruta con defectos superficiales localizados (manchas, cicatrices, "
        "raspaduras o magulladuras leves), aún sin deterioro extenso."
    ),
    "Mala": (
        "Fruta con deterioro marcado: pudrición, áreas oscuras extensas, "
        "moho visible o daño estructural severo."
    ),
}

# Mapeo de nombres de especies de Kaggle (inglés) a nombre canónico en español
KAGGLE_SPECIES_MAP: dict[str, str] = {
    "apple": "manzana",
    "banana": "banano",
    "guava": "guayaba",
    "lemon": "limon",
    "lime": "limon",
    "orange": "naranja",
    "pomegranate": "granada",
}

# Carpetas esperadas dentro de data/external/ (Fruit Quality Classification / FruitNet)
KAGGLE_QUALITY_BUCKETS: dict[str, str] = {
    "Good Quality_Fruits": "Buena",
    "Bad Quality_Fruits": "Mala",
}
KAGGLE_MIXED_FOLDERS: tuple[str, ...] = ("Mixed Qualit_Fruits", "Mixed Quality_Fruits")

# Parámetros de preprocesamiento e imágenes
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAX_WORKING_SIDE_PX = 1024
JPEG_QUALITY = 90
DL_IMAGE_SIZE = 224
CLASSICAL_IMAGE_SIZE = 128

# Categorías relativas de tamaño (basadas en diámetro normalizado respecto al lado mayor max(W, H))
SIZE_CATEGORIES: tuple[str, ...] = ("Pequeño", "Mediano", "Grande")
