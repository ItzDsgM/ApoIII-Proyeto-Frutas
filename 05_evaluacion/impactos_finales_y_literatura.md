# Fase 5 CRISP-DM — Análisis Final de Impactos (ex-post), Domain Shift y Literatura Científica

Este documento complementa el análisis inicial de la Fase 1 ([`01_comprension_negocio/etica_e_impactos.md`](../01_comprension_negocio/etica_e_impactos.md)) con el **análisis final cuantificado (ex-post)** basado exclusivamente en los artefactos congelados y métricas reales del conjunto de prueba (`n = 194`) y las **307 referencias geométricas** propias.

---

## 1. Comparación de impactos: supuesto inicial (Fase 1) vs. evidencia final (Fase 5)

| Dimensión | Supuesto inicial (Fase 1, ex-ante) | Evidencia cuantificada final (Fase 5, ex-post) | Brecha y acción correctiva |
|---|---|---|---|
| **Económica y operativa** | Clasificación rápida en CPU estándar sin GPU dedicada, evitando rechazar fruta aprovechable. | **Latencia medida:** `35.44 ms/img` (~`28.2 FPS` secuenciales en CPU de 12 núcleos), peso de modelo `9.34 MB`. Globalmente `MobileNetV2` alcanza `94.33 %` de exactitud (`Macro-F1 = 0.9382`). | **Riesgo de falsos positivos `Mala → Buena`:** `7/87` (`8.05 %`) en prueba global, concentrados en fotos propias (`5/7` `Mala` propias). No debe operar sin supervisión humana. |
| **Social y *human-in-the-loop*** | Herramienta de apoyo transparente para operarios y pequeños productores, sin reemplazar criterio sanitario. | La interfaz `06_despliegue/app.py` expone probabilidades por clase, máscara de segmentación y contexto por especie. Se recomienda derivar a **revisión humana obligatoria** cualquier predicción con $\max_c P(y=c \mid I) < 0.75$. | En `Propia` (`n=61`), la clase `Regular` obtuvo solo `1/3` (`33.3 %` recall) frente a `17/17` (`100 %`) en `Mixed`. El operario debe inspeccionar manualmente defectos sutiles. |
| **Ambiental y eficiencia computacional** | Reducir desperdicio de fruta `Regular` aprovechable y minimizar consumo energético y de almacenamiento. | Extractor `MobileNetV2` congelado (`164,355` parámetros entrenables de `2.39M` totales; `352.4 s` de entrenamiento en CPU). Limpieza y redimensionado de fotos propias redujo el peso de `374.1 MB` a `16.84 MB` (`-95.5 %`) con `0` metadatos EXIF. | El módulo geométrico de tamaño logra `100 %` de cobertura (`307/307`) con `MAE = 0.0222` (`Mediana = 0.0067`) y `92.83 %` de exactitud categórica (`Macro-F1 = 0.9311`), evitando desperdicio por errores de calibre. |
| **Global, privacidad y dominio** | Portabilidad local entre especies colombianas (`mango`, `tomate_arbol`) e internacionales (FruitNet) cumpliendo privacidad (PI1). | Procesamiento 100 % offline sin envío a servidores externos. Se identificó **sesgo de dominio (*domain shift* y *shortcut learning*)** entre recortes `Mixed` (`52×55 px`, `100 % Regular`) y capturas propias de alta resolución. | Exigir captura sobre fondo uniforme mate, advertir explícitamente en `app.py` sobre `tomate_arbol` (`0` ejemplos `Regular`/`Mala` en entrenamiento) y planificar recolección balanceada futura. |

---

## 2. Diagnóstico cuantitativo de *Domain Shift* y *Shortcut Learning* (Opción B)

En lugar de reentrenar los pesos congelados, se documenta rigurosamente el origen de los **11 errores** de `MobileNetV2 (Transfer)` sobre el conjunto de prueba (`n = 194`):

| Origen (`source`) | `n` | Exactitud | Macro-F1 (clases presentes) | Errores observados |
|---|---:|---:|---:|---|
| `FruitNet_Kaggle` | 116 | `97.41 %` | `0.9700` | `3` (`1 Buena→Mala`, `2 Mala→Buena`) |
| `FruitNet_Mixed_Segmentada` | 17 | `100.00 %` | `1.0000` | `0` (todos los recortes `Mixed` son `Regular`, mediana `52×55 px`) |
| `Propia` | 61 | `86.89 %` | `0.5901` | `8` (`5 Mala→Buena`, `2 Regular→Buena`, `1 Buena→Regular`) |

### Causas raíz identificadas
1. **Atajo por resolución/origen (*Shortcut Learning*) en `Regular`:** Los `84` recortes validados de `Mixed Qualit_Fruits` pertenecen en su totalidad a la clase `Regular` (`0 Buena`, `0 Mala`) y tienen resolución mediana de `52×55 px` (interpolados a `224×224` para `MobileNetV2`). En contraste, las fotos propias tienen hasta `1024 px` de lado mayor y solo aportan `15` ejemplos `Regular`. El modelo aprendió parcialmente la correlación espuria *"textura reescalada de baja resolución = Regular"*, acertando `17/17` en `Mixed` pero fallando `2/3` en `Regular` propio (`limon_043_regular.jpg`, `mango_039_regular.jpg`).
2. **Desbalanceo por especie en capturas propias:** `tomate_arbol` (`58`) y `manzana` propia (`29`) solo tienen ejemplos `Buena`; en `banano` (`2` `Mala→Buena`) y `naranja` (`3` `Mala→Buena`) de prueba propia, los defectos localizados o fondos oscuros redujeron la sensibilidad hacia `Mala`.
3. **Diferencia en estimación de tamaño sobre escenas multifruto:** Sobre las `307` referencias propias, la mediana del error absoluto es muy baja (`0.0067`), pero en imágenes con múltiples frutos próximos (`manzana_024..029`, `naranja_069..071`), `estimate_fruit_size` fusiona componentes adyacentes, explicando el `RMSE = 0.0643` frente a una mediana de `0.0067`.

---

## 3. Contraste con literatura científica (formato IEEE)

| Referencia científica (IEEE) | Contribución o hallazgo clave | Contraste directo con este proyecto |
|---|---|---|
| V. Meshram y K. Patil, "FruitNet: Indian Fruits Image Dataset with Quality for Machine Learning Applications," *Data in Brief*, vol. 39, 107686, 2021. | Introduce *FruitNet* (`Good`, `Bad`, `Mixed`) destacando variabilidad de iluminación y fondos en frutas tropicales e indias. | Nuestros experimentos confirman que `Good` vs. `Bad` en FruitNet alcanza `97.41 %` con MobileNetV2, pero la carpeta `Mixed` requiere segmentación previa por instancia y puede introducir sesgo si solo aporta una clase (`Regular`). |
| M. Sandler, A. Howard, M. Zhu, A. Zhmoginov y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *Proc. IEEE/CVF CVPR*, 2018, pp. 4510–4520. | Propone bloques residuales invertidos y convoluciones separables en profundidad para inferencia móvil eficiente. | Con extractor congelado (`164k` params entrenables), `MobileNetV2` supera ampliamente a ML clásico (`0.9382` vs. `0.7927` Macro-F1) manteniendo `35.44 ms/img` en CPU y `9.34 MB` de peso. |
| S. Kornblith, J. Shlens y Q. V. Le, "Do Better ImageNet Models Transfer Better?," *Proc. IEEE/CVF CVPR*, 2019, pp. 2661–2671. | Demuestra que representaciones preentrenadas en ImageNet transfieren con mayor generalización y menor sobreajuste en datasets pequeños. | Con `621` muestras de entrenamiento, ML clásico sobre `d=408` features sufre sobreajuste severo (brecha `0.21–0.26`), mientras `MobileNetV2` reduce la brecha `Train−Test` a `0.0084`. |
| R. Geirhos et al., "Shortcut Learning in Deep Neural Networks," *Nature Machine Intelligence*, vol. 2, pp. 665–673, 2020. | Formaliza cómo las redes explotan señales espurias (fondo, textura o resolución) que fallan bajo cambio de dominio (*OOD*). | Explica directamente por qué `MobileNetV2` logra `100 %` en `Mixed` (`52×55 px`) pero `0.5901` de Macro-F1 en `Propia`: la resolución/origen actuó como atajo para la clase `Regular`. |
| N. Dalal y B. Triggs, "Histograms of Oriented Gradients for Human Detection," *Proc. IEEE/CVF CVPR*, 2005, pp. 886–893. | Define descriptores de gradiente local normalizados por bloques (`L2-Hys`). | Se reutilizó la formulación HOG compacta (`324` dims) dentro del vector de `408` características clásicas (`src/features.py`), útil como base interpretable pero insuficiente frente a variaciones de iluminación y textura fina. |
