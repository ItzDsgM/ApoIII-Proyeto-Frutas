# Clasificación de Calidad Visual y Estimación de Tamaño Relativo de Frutas

Proyecto Final — **Algoritmos y Programación III** (Semestre 2026-2)  
Facultad de Ingeniería, Diseño y Ciencias Aplicadas · Ingeniería de Sistemas · **Universidad Icesi**

Sistema automático basado en visión por computadora para clasificar frutas individuales en **3 categorías de calidad visual** (`Buena`, `Regular`, `Mala`) y estimar su **tamaño relativo** (diámetro normalizado invariante a orientación y categoría `Pequeño / Mediano / Grande`), desarrollado bajo la metodología **CRISP-DM**.

---

## Estructura del repositorio (CRISP-DM)

```text
APO-III-Proyecto-Frutas/
├── 01_comprension_negocio/        # Fase 1: formulación matemática (PI3), ética ACM (PI1), impactos (PI2) y flujo CRISP-DM
├── 02_comprension_datos/          # Fase 2: notebook EDA ejecutado con gráficas y galería de muestras
├── 03_preparacion_datos/          # Fase 3: limpieza EXIF, segmentación de Mixed, splits anti-fuga (dHash)
├── 04_modelado/                   # Fase 4: entrenamiento reproducible + notebooks de ML clásico y Deep Learning
├── 05_evaluacion/                 # Fase 5: notebook de evaluación, métricas JSON/CSV y 7 figuras vectoriales (.svg/.png)
├── 06_despliegue/                 # Fase 6: aplicación Streamlit offline + modelo ganador (best_quality_model.pt)
├── src/                           # Módulos Python limpios (PEP 8) y documentados
├── data/                          # Fotos propias limpias (16.8 MB, sin EXIF), recortes Mixed y CSVs de anotaciones
├── tests/                         # Suite pytest (preprocesamiento, tamaño, inferencia y Streamlit AppTest)
└── docs/                          # Reservado para informe final IEEE (<=7 págs) y video (<=10 min)
```

---

## Datos y preparación

Se combinaron tres fuentes controladas en **971 imágenes** agrupadas con hash perceptual `dHash` y `source_group_id` para garantizar **cero fuga de datos** entre `train` (`621`), `val` (`156`) y `test` (`194`):

| Origen | Buena | Regular | Mala | Total |
|---|---:|---:|---:|---:|
| Fotos propias validadas (`6` especies, sin EXIF, ≤ 1024 px) | 255 | 15 | 37 | **307** |
| `Mixed Qualit_Fruits` segmentadas (CIELAB + Otsu + Watershed) | 0 | 84 | 0 | **84** |
| Kaggle *Fruit Quality Classification* (`Good` / `Bad`, 4 especies) | 180 | 0 | 400 | **580** |
| **Total combinado** | **435** | **99** | **437** | **971** |

- **Especies cubiertas (6):** `banano`, `limon`, `mango`, `manzana`, `naranja`, `tomate_arbol`.
- **Privacidad (PI1):** se eliminaron el 100 % de metadatos EXIF en las 307 fotos propias y se redujo su peso de `374.1 MB` a `16.84 MB`.

---

## Resultados principales

### 1. Clasificación de calidad (conjunto de prueba reservado, `n = 194`)

Todos los modelos se evaluaron sobre exactamente la misma partición de prueba. La selección del modelo de despliegue se realizó **estrictamente por `Val Macro-F1`**.

| Modelo | Familia | Train F1 | Val F1 | **Test Macro-F1** | **Test Bal. Acc** | **Test Acc** | Brecha (Train−Test) | Tiempo entr. | Latencia CPU | Peso |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MobileNetV2 (Transfer)** | Deep Learning | 0.9466 | **0.9013** | **0.9382** | **0.9322** | **0.9433** | **0.0084** | 352.4 s | 35.4 ms | 9.34 MB |
| **CNN simple (desde cero)** | Deep Learning | 0.8021 | 0.8342 | 0.8013 | 0.8389 | 0.8351 | 0.0008 | 220.7 s | 30.1 ms | 0.40 MB |
| **XGBoost (Clásico)** | ML Clásico | 1.0000 | 0.7980 | 0.7927 | 0.7644 | 0.8557 | 0.2073 | 78.9 s | 28.6 ms | 0.16 MB |
| **SVM (Clásico)** | ML Clásico | 0.9988 | 0.7757 | 0.7687 | 0.7452 | 0.8299 | 0.2301 | 7.7 s | 29.1 ms | 0.86 MB |
| **Random Forest (Clásico)** | ML Clásico | 0.9988 | 0.7650 | 0.7374 | 0.7042 | 0.8093 | 0.2614 | 12.4 s | 70.4 ms | 1.33 MB |
| Baseline (Estratificada) | Línea base | 0.3666 | 0.3385 | 0.2484 | 0.2529 | 0.3402 | 0.1182 | <0.1 s | 0.1 ms | — |
| Baseline (Mayoritaria) | Línea base | 0.2072 | 0.2065 | 0.2064 | 0.3333 | 0.4485 | 0.0008 | <0.1 s | 0.1 ms | — |

- **Desempeño por clase de MobileNetV2 (Test):** `Buena` F1 = `0.9392` (`85/87`), `Regular` F1 = `0.9231` (`18/20`), `Mala` F1 = `0.9524` (`80/87`).
- **Observación crítica de dominio:** en las 61 fotos propias de prueba, MobileNetV2 obtuvo `86.89 %` de exactitud (`Macro-F1 = 0.5901` por el soporte reducido de `Regular` en fotos propias frente a los recortes de menor resolución de `Mixed Qualit_Fruits`). Esta brecha de dominio se documenta abiertamente en `05_evaluacion/05_evaluacion_y_errores.ipynb`.

### 2. Estimación de tamaño relativo

Se emplea segmentación combinada **CIELAB + saturación HSV + Otsu + selección por centralidad** y la métrica invariante a rotaciones de 90° `max(bw, bh) / max(W, H)`:

- **Evaluación sobre 204 referencias geométricas propias:** cobertura **204/204 (100 %)**, **MAE = `0.0124`**, **Mediana = `0.0060`**, **RMSE = `0.0192`**.
- **Cobertura sobre el dataset combinado (971 imágenes):** **969/971 (99.8 %)** (`307/307` propias, `580/580` Kaggle, `82/84` recortes `Mixed`).

---

## Instalación y ejecución rápida

### 1. Instalar dependencias (Python 3.12)
```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ejecutar la interfaz web (funciona inmediatamente sin descargar nada)
```bash
streamlit run 06_despliegue/app.py
```

### 3. Ejecutar pruebas automatizadas
```bash
pytest -v
```

### 4. Reproducir entrenamiento completo (opcional — requiere descargar FruitNet en `data/external/`)
```bash
python 03_preparacion_datos/construir_splits.py
python 04_modelado/entrenar_modelos.py
```

---

## Fuentes y licencias

- Kaggle *Fruit Quality Classification*: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
- FruitNet original (CC BY 4.0): Meshram & Patil (2021), DOI `10.1016/j.dib.2021.107686`, https://data.mendeley.com/datasets/b6fftwbr2v/2
- MobileNetV2 preentrenada (`IMAGENET1K_V2`): PyTorch / TorchVision
- Código de Ética ACM: https://www.acm.org/code-of-ethics
