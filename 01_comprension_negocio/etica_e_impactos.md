# Ética Profesional (PI1) y Análisis de Impactos (PI2)

## 1. Dilemas éticos y relación con el Código de Ética ACM (PI1)

| Situación o decisión ética | Principio del Código ACM | Medida implementada en el proyecto |
|---|---|---|
| **Privacidad en fotos y captura en vivo por cámara** (posibles metadatos GPS/dispositivo, rostros o entorno doméstico en cámara web) | **1.6 Respetar la privacidad** | `03_preparacion_datos/preparar_datos.py` elimina el 100 % de metadatos EXIF de las 307 fotos propias (verificado automáticamente); la app procesa capturas de cámara en memoria local sin almacenarlas ni enviarlas a servidores externos. |
| **Consentimiento y trazabilidad de autoría de datos/modelos** | **1.5 Respetar el trabajo requerido para producir nuevas ideas** | Se atribuyen explícitamente FruitNet (CC BY 4.0, Meshram & Patil 2021), Kaggle *Fruit Quality Classification* y pesos oficiales MobileNetV2 de TorchVision. Las fotos propias provienen exclusivamente del equipo. |
| **Sesgos por desbalanceo de clases y especies** (`Regular` minoritaria; `tomate_arbol` sin ejemplos `Mala`) | **1.4 Ser justo y no discriminar**; **1.3 Ser honesto y confiable** | Se utiliza Macro-F1 y Exactitud Balanceada (no solo Accuracy), entropía cruzada ponderada, segmentación de `Mixed Qualit_Fruits` para reforzar `Regular`, y desglose público de errores por especie y por origen. |
| **Confundir inspección visual con inocuidad alimentaria** (una fruta visualmente sana puede tener daño interno o contaminación invisible) | **1.2 Evitar el daño**; **2.5 Realizar evaluaciones exhaustivas de riesgos** | La interfaz y la documentación establecen explícitamente que la salida evalúa únicamente apariencia externa y requiere supervisión humana; nunca emite certificación sanitaria ni decisión automática de consumo. |
| **Riesgo de fuga de datos y sobreestimación de resultados** | **1.3 Ser honesto y confiable**; **2.1 Esforzarse por lograr alta calidad** | Agrupación perceptual por `dHash` y `source_group_id` antes de `StratifiedGroupKFold`, separación estricta `train / val / test`, selección del modelo ganador exclusivamente con `val` y publicación de errores reales y brecha de sobreajuste. |

## 2. Análisis inicial de impactos (Fase 1, ex-ante) — Matriz global, social, ambiental y económica (PI2)

> **Alcance temporal:** Esta matriz recoge la evaluación **inicial (ex-ante)** realizada durante la Fase 1 bajo supuestos previos al entrenamiento. El **análisis final cuantificado (ex-post)**, contrastado con las métricas de prueba, latencia real en CPU, reducción de peso y errores observados por origen y especie, se encuentra documentado en [`05_evaluacion/impactos_finales_y_literatura.md`](../05_evaluacion/impactos_finales_y_literatura.md).

### Supuestos operativos iniciales (ex-ante)
- Se asumió que las fotos propias y las fuentes públicas (`FruitNet_Kaggle` + `Mixed Qualit_Fruits`) compartirían distribuciones suficientemente compatibles tras recorte de primer plano.
- Se estableció como restricción de diseño operar exclusivamente en CPU estándar sin GPU dedicada ni transmisión de imágenes a servicios en la nube.

| Dimensión | Impactos positivos potenciales | Riesgos o impactos negativos | Estrategias concretas de mitigación |
|---|---|---|---|
| **Social** | Estandarización objetiva en mercados y cooperativas; herramienta educativa accesible para pequeños agricultores. | Desplazamiento o desconfianza de operarios; percepción de vigilancia si se despliegan cámaras en zonas de trabajo. | Diseñar el sistema como apoyo *human-in-the-loop* (sin decisiones autónomas irreversibles), enfocar la cámara exclusivamente hacia la bandeja del producto y procesar localmente sin grabar personas. |
| **Económica** | Reducción del tiempo de clasificación manual (latencia de milisegundos en CPU estándar sin GPU costosa) y mejor clasificación por lotes. | Falsos negativos (`Buena`/`Regular` clasificadas como `Mala`) generan pérdidas directas al productor; falsos positivos afectan la confianza del comprador. | Reportar explícitamente matriz de confusión por clase, mostrar probabilidades en la interfaz para que casos dudosos pasen a revisión humana y emplear hardware de bajo costo (CPU). |
| **Ambiental** | Reducir el desperdicio de alimentos separando fruta `Regular` aprovechable (jugos, pulpas, venta secundaria) de fruta `Mala`. | Consumo energético por entrenamiento de redes profundas y acumulación de residuos si el modelo rechaza fruta sana en exceso. | Usar arquitecturas eficientes (CNN liviana + MobileNetV2 con extractor congelado, evitando reentrenar millones de parámetros), registrar tiempos de entrenamiento, peso en MB y latencia en `05_evaluacion/metricas/resultados_completos.json`, y reducir fotos propias de 374 MB a 16.8 MB. |
| **Global** | Aplicabilidad en distintas regiones agrícolas y mercados emergentes usando cámaras de celular o webcams económicas. | Variedades locales, iluminación y estándares regulatorios (calidad comercial, protección de datos como Ley 1581 en Colombia o GDPR) varían entre países. | Integrar especies locales colombianas (como `tomate_arbol` y `mango`) junto con FruitNet (India), documentar límites de dominio por especie y exigir calibración local antes de cualquier uso productivo. |

*Referencia ética:* ACM Code of Ethics and Professional Conduct (https://www.acm.org/code-of-ethics).
