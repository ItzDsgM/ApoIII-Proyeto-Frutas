# Datos del proyecto

Este directorio separa claramente las **fuentes públicas**, las **fotos propias validadas por el equipo** y los **índices de anotaciones** reproducibles.

## Fuentes

1. **Dataset público (Kaggle — Enunciado sec. 2.1):**
   - *Fruit Quality Classification* (basado en FruitNet, licencia CC BY 4.0):
     https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
   - Referencia original: Meshram & Patil (2021), *FruitNet: Indian Fruits Image Dataset with Quality for Machine Learning Applications*, Data in Brief, DOI: `10.1016/j.dib.2021.107686`.
   - Ubicación local (excluida de Git en `.gitignore`): `data/external/` con las carpetas `Good Quality_Fruits/`, `Bad Quality_Fruits/` y `Mixed Qualit_Fruits/`.
   - **Selección de 4 especies de Kaggle:** de las 6 especies originales de FruitNet (`Apple`, `Banana`, `Guava`, `Lime/Lemon`, `Orange`, `Pomegranate`), se seleccionaron únicamente las **4 especies compartidas con la recolección local** (`Apple` → `manzana`, `Banana` → `banano`, `Lime/Lemon` → `limon`, `Orange` → `naranja`). Se excluyeron `Guava` y `Pomegranate` por ausencia de recolección propia equivalente; en su lugar, el inventario propio incorpora dos frutas locales frecuentes en el mercado colombiano (`mango` y `tomate_arbol`) para evaluar el comportamiento del modelo ante especies sin respaldo en Kaggle.
   - **Nota sobre `Mixed Qualit_Fruits`:** contiene escenas con múltiples frutas de distintas calidades; nunca se asigna automáticamente a una clase. Se segmentan ejemplares individuales con `03_preparacion_datos/segmentar_mixed.py` y se etiquetan manualmente antes de incorporarlos al conjunto de datos. Los 84 recortes validados quedaron todos en `Regular` (`52×55 px` de resolución mediana), lo cual se analiza explícitamente como fuente potencial de *shortcut learning* por resolución/origen en la Fase 5.

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

- `own_annotations.csv`: inventario maestro unificado de **calidad y tamaño** para las **307/307** fotos propias (`image_path`, `species`, `quality`, `width`, `height`, `file_bytes`, `sha256`, `original_filename`, `original_sha256`, `source`, `group_id`, `reviewer`, `reference_normalized_diameter`, `size_category`).
- `size_references.csv`: referencias geométricas de tamaño de las **307/307** fotos propias con coordenadas verificadas (`reference_bbox_x1..y2` en las 103 incorporaciones recientes), `reference_normalized_diameter` ($\max(b_w, b_h) / \max(W, H)$) y `reference_size_category` (`Pequeño`, `Mediano`, `Grande` según `DEFAULT_SIZE_THRESHOLDS = (0.30, 0.55)`).
- `excluded_own_photos.csv`: registro de duplicados o descartes documentados.
- `mixed_crops_annotations.csv`: recortes individuales segmentados de `Mixed Qualit_Fruits` y validados manualmente, vinculados a su foto fuente (`group_id`) para evitar fuga de datos.
- `fruitnet_splits.csv`: particiones reproducibles (`train`/`val`/`test`) del conjunto combinado.

## Protocolo de compartición de anotaciones entre grupos (Sección 2.3)

Para facilitar la consolidación e intercambio reproducible de etiquetas de calidad y tamaño entre distintos grupos de trabajo:

1. **Formato tabular estándar (`own_annotations.csv`):** cada lote compartido se exporta en codificación UTF-8 con separación por comas e incluye obligatoriamente las columnas `image_path`, `species`, `quality` (`Buena` / `Regular` / `Mala`), `reference_normalized_diameter`, `size_category` (`Pequeño` / `Mediano` / `Grande`), `sha256`, `source`, `group_id` y `reviewer`.
2. **Trazabilidad de origen y revisor (`source` y `reviewer`):**
   - `source` identifica la procedencia del lote (por ejemplo, `"Propia"` para las capturas del equipo local o `"Grupo_<ID>"` para lotes recibidos de otro equipo).
   - `reviewer` registra el identificador del anotador o equipo responsable de la verificación visual de calidad y tamaño.
3. **Flujo general de integración sin fuga de datos:**
   - Al recibir un lote externo en formato estándar, se verifica el esquema y la coherencia de umbrales (`size_category` frente a `DEFAULT_SIZE_THRESHOLDS = (0.30, 0.55)`).
   - Se ejecuta `03_preparacion_datos/preparar_datos.py` para desduplicar por `sha256`, eliminar metadatos EXIF y preservar las etiquetas unificadas.
   - Se consolida en `03_preparacion_datos/construir_splits.py`, donde `group_id` + hash perceptual `dHash` garantizan que fotografías repetidas o de una misma escena nunca queden divididas entre `train`, `val` y `test`.
