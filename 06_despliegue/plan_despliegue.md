# Fase 6 CRISP-DM — Plan de Despliegue y Trabajo Futuro

## 1. Arquitectura y artefactos de despliegue

El sistema opera de forma **100 % local y offline** (sin conexión a internet ni envío de imágenes a servidores externos), cumpliendo el principio ético de privacidad (ACM 1.6):

- **Diagrama vectorial de arquitectura:** [`06_despliegue/arquitectura_software.svg`](arquitectura_software.svg).
- **Aplicación interactiva:** [`06_despliegue/app.py`](app.py) construida con Streamlit.
- **Modelo ganador congelado:** [`06_despliegue/models/best_quality_model.pt`](models/best_quality_model.pt) (`MobileNetV2`, `9.34 MB`, entrada `224×224`, seleccionado estrictamente por `Val Macro-F1 = 0.9013`).
- **Metadatos de despliegue:** [`06_despliegue/models/model_metadata.json`](models/model_metadata.json).

---

## 2. Requisitos mínimos de operación y rendimiento medido

- **Hardware mínimo:** procesador x86_64/ARM estándar de 2+ núcleos (sin GPU requerida), 2 GB de RAM disponibles, ~150 MB de almacenamiento para entorno Python + dependencias y cámara web o archivo de imagen (`.jpg`, `.jpeg`, `.png`, `.webp`).
- **Rendimiento medido en CPU:** latencia promedio de **`35.44 ms/imagen`** (~28 FPS secuenciales) incluyendo recorte de primer plano e inferencia de calidad.
- **Software:** Python 3.12+ y dependencias fijadas en `requirements.txt`.

---

## 3. Protocolo operativo de captura y flujo *Human-in-the-Loop*

Para replicar las condiciones donde el sistema alcanza mayor precisión tanto en calidad como en estimación geométrica de tamaño:

1. **Encuadre individual y fondo uniforme:** fotografiar **una sola fruta centrada** sobre fondo liso y mate (tela blanca o mesón neutro uniforme) sin que el contorno toque los bordes de la imagen (`touches_border = False`).
2. **Iluminación difusa:** evitar sombras proyectadas intensas y reflejos especulares directos que distorsionen el canal `L*` o la máscara de segmentación.
3. **Regla de revisión humana (*Human-in-the-Loop*):**
   - Si la confianza del modelo es inferior al umbral operativo ($\max_c P(y = c \mid I) < 0.75$), `app.py` emite automáticamente una alerta solicitando inspección visual directa del operario.
   - Si el usuario selecciona una especie en el panel lateral (`selected_species`), la interfaz muestra el desempeño real de prueba y advertencias transparentes de dominio (por ejemplo, que `tomate_arbol` solo contó con muestras `Buena` en entrenamiento).
   - La salida del prototipo evalúa **exclusivamente apariencia externa** y nunca sustituye una certificación bromatológica o sanitaria.

---

## 4. Verificación continua y empaquetado local

Antes de cada entrega o actualización del entorno:
```bash
.venv/bin/pytest -v
.venv/bin/streamlit run 06_despliegue/app.py
```
La suite `tests/` valida automáticamente la limpieza EXIF, la invariancia a rotaciones de 90° del estimador de tamaño, la completitud de `307/307` anotaciones propias, el vector clásico de `408` dimensiones, la carga offline de `best_quality_model.pt` y el renderizado integral de `app.py` mediante `AppTest`.

---

## 5. Trabajo futuro recomendado

1. **Mitigación directa del *domain shift* (siguiente iteración CRISP-DM):** recolectar muestras propias reales de clases `Regular` y `Mala` en `tomate_arbol`, `manzana` y `limon`, además de clasificar recortes `Buena` y `Mala` dentro de `Mixed Qualit_Fruits` para eliminar el atajo resolución-clase.
2. **Calibración física de tamaño:** incorporar un marcador fiduciario opcional (por ejemplo, moneda o tarjeta de referencia en el borde) para convertir el diámetro normalizado adimensional en milímetros reales cuando el caso de uso lo requiera.
3. **Despliegue embebido en borde (*edge*):** exportar el modelo a ONNX/TorchScript cuantizado (`INT8`) sobre Raspberry Pi 4/5 con cámara fija sobre banda o bandeja de inspección iluminada.
