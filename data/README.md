# Datos del proyecto

Este directorio separa claramente las **fuentes públicas**, las **fotos propias validadas por el equipo** y los **índices de anotaciones** reproducibles.

## Fuentes

1. **Dataset público (Kaggle — Enunciado sec. 2.1):**
   - *Fruit Quality Classification* (basado en FruitNet, licencia CC BY 4.0):
     https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
   - Referencia original: Meshram & Patil (2021), *FruitNet: Indian Fruits Image Dataset with Quality for Machine Learning Applications*, Data in Brief, DOI: `10.1016/j.dib.2021.107686`.
   - Ubicación local (excluida de Git en `.gitignore`): `data/external/` con las carpetas `Good Quality_Fruits/`, `Bad Quality_Fruits/` y `Mixed Qualit_Fruits/`.
   - **Nota sobre `Mixed Qualit_Fruits`:** contiene escenas con múltiples frutas de distintas calidades; nunca se asigna automáticamente a una clase. Se segmentan ejemplares individuales con `03_preparacion_datos/segmentar_mixed.py` y se etiquetan manualmente antes de incorporarlos al conjunto de datos.

2. **Recolección propia del equipo (Enunciado sec. 2.2–2.3):**
   - Ubicación: `data/own_photos/<especie>/<Buena|Regular|Mala>/` (`banano`, `limon`, `mango`, `manzana`, `naranja`, `tomate_arbol`).
   - Las imágenes de trabajo versionadas pasan por `03_preparacion_datos/preparar_datos.py --apply`:
     - Deduplicación binaria exacta por SHA-256.
     - Corrección de orientación EXIF y **eliminación total de metadatos EXIF** (privacidad, PI1).
     - Redimensionado proporcional a lado máximo `1024 px` (Lanczos, JPEG calidad 90) para mantener el repositorio ligero sin perder detalle visual.
     - Nombrado uniforme `{especie}_{nnn}_{calidad}.jpg`.

## Esquema de clases de calidad

| Clase | Criterio operativo visual |
|---|---|
| `Buena` | Fruta fresca e íntegra en la superficie visible; color normal y sin golpes ni manchas significativas. |
| `Regular` | Defectos superficiales localizados (manchas, cicatrices, raspaduras o golpes leves), sin deterioro extenso. |
| `Mala` | Deterioro marcado: pudrición, áreas oscuras extensas, moho visible o daño estructural severo. |

> Las categorías describen **calidad visual externa**; no certifican inocuidad ni estado interno del alimento.

## Archivos de anotaciones (`data/annotations/`)

- `own_annotations.csv`: inventario maestro de fotos propias (ruta, especie, calidad, dimensiones, SHA-256, `group_id` y revisor). Listo para compartir anotaciones con el curso.
- `excluded_own_photos.csv`: registro de duplicados o descartes documentados.
- `mixed_crops_annotations.csv`: recortes individuales segmentados de `Mixed Qualit_Fruits` y validados manualmente, vinculados a su foto fuente (`group_id`) para evitar fuga de datos.
- `fruitnet_splits.csv`: particiones reproducibles (`train`/`val`/`test`) del conjunto combinado.
