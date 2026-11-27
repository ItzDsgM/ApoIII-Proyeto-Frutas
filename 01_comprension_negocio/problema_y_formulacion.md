# Fase 1 — Comprensión del Negocio y Formulación Matemática (PI3)

## 1. Contexto y problema de ingeniería

En mercados locales y pequeñas agroindustrias, la inspección visual de frutas frescas se realiza de forma manual, lo cual es lento, subjetivo y propenso a inconsistencias entre operarios. Esto incrementa pérdidas económicas por rechazos erróneos y desperdicio de alimentos aprovechables.

Se trata de un **problema complejo de ingeniería** porque intervienen simultáneamente:
- **Alta variabilidad visual intraclase e interclase:** 6 especies distintas (`banano`, `limon`, `mango`, `manzana`, `naranja`, `tomate_arbol`), iluminación heterogénea, fondos claros y oscuros, y cambios de maduración natural que pueden confundirse con defectos.
- **Desbalance de clases realista:** la categoría intermedia (`Regular`) es mucho menos frecuente en fuentes públicas que los extremos (`Buena` y `Mala`), y la carpeta `Mixed Qualit_Fruits` mezcla múltiples frutas por foto.
- **Restricciones de hardware y despliegue local:** inferencia en CPU sin GPU dedicada, preservando privacidad y baja latencia en una interfaz interactiva.

## 2. Formulación matemática

### 2.1. Clasificación de calidad visual
Sea una imagen RGB orientada $I \in \{0, \dots, 255\}^{H \times W \times 3}$ de una fruta individual. Se busca una función $f_\theta : \mathbb{R}^{H \times W \times 3} \rightarrow \Delta^2$ que estime la distribución de probabilidad sobre el conjunto de clases ordenadas $\mathcal{Y} = \{\text{Buena}, \text{Regular}, \text{Mala}\}$ ($K = 3$):

$$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} P_\theta(y = c \mid \phi(I))$$

donde $\phi(I)$ corresponde a:
- Un vector de características diseñadas $\mathbf{x} \in \mathbb{R}^{408}$ ($48$ histogramas HSV + $18$ momentos CIELAB + $6$ indicadores de manchas oscuras locales + $12$ textura Sobel/Laplaciano + $324$ HOG) para los modelos clásicos (SVM, Random Forest, XGBoost), o
- Un tensor normalizado de píxeles tras recorte del primer plano para las arquitecturas convolucionales (CNN desde cero y MobileNetV2 congelada).

Para compensar el desbalance entre clases se optimiza la **entropía cruzada ponderada**:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} \log P_\theta(y_i \mid I_i), \qquad w_c = \frac{N}{K \cdot N_c}$$

### 2.2. Segmentación multifruto y estimación de tamaño relativo
Para segmentar frutas respecto al fondo, se transforma la imagen al espacio perceptual **CIELAB** $(L^*, a^*, b^*)$ y se calcula la distancia ponderada al color mediano del marco exterior $(\tilde{L}, \tilde{a}, \tilde{b})$:

$$d(u, v) = \sqrt{w_L \,(L^*(u, v) - \tilde{L})^2 + (a^*(u, v) - \tilde{a})^2 + (b^*(u, v) - \tilde{b})^2}$$

con $w_L \in [0.30, 0.35]$ para atenuar sombras proyectadas. Tras umbralización automática de Otsu y operaciones morfológicas, se obtiene la caja delimitadora $(b_w, b_h)$ del fruto principal y se define el **diámetro normalizado invariante a rotaciones de 90°**:

$$s(I) = \frac{\max(b_w, b_h)}{\max(W, H)} \in (0, 1]$$

A diferencia de normalizar por el ancho $W$, dividir por $\max(W, H)$ garantiza $s(I) = s(\text{rot}_{90^\circ}(I))$ y evita que una misma fruta parezca el doble de grande en orientación vertical frente a horizontal. El error geométrico frente a las $M$ cajas de referencia manuales se mide con:

$$\text{MAE} = \frac{1}{M}\sum_{j=1}^{M} \left| s_j^{\text{ref}} - \hat{s}_j \right|, \qquad \text{RMSE} = \sqrt{\frac{1}{M}\sum_{j=1}^{M} \left( s_j^{\text{ref}} - \hat{s}_j \right)^2}$$

### 2.3. Métricas de desempeño y control de sobreajuste
Dado el desbalance de la clase `Regular`, la métrica principal de selección y evaluación es **Macro-F1** (promedio aritmético no ponderado de $F_{1,c} = \frac{2 P_c R_c}{P_c + R_c}$ sobre las 3 clases), complementada con Exactitud Balanceada, Exactitud global, matrices de confusión y la brecha de generalización $\Delta_{\text{overfit}} = \text{Macro-F1}_{\text{train}} - \text{Macro-F1}_{\text{test}}$.

## 3. Criterios operativos de clases

- **Buena:** fruta fresca e íntegra en la superficie visible, sin golpes ni manchas relevantes.
- **Regular:** defectos superficiales localizados (manchas, cicatrices, raspaduras o magulladuras leves) sin pudrición extensa.
- **Mala:** deterioro marcado, pudrición, moho visible o áreas oscuras extensas.
