# Pipeline de Machine Learning: Dataset Iris

> **Contexto:** Esta carpeta contiene la práctica correspondiente a la Lecture 3 del curso universitario de **Fundamentos de Aprendizaje Automático**.

## 🎯 Objetivo de la Práctica

El objetivo principal de esta etapa es diseñar y ejecutar las primeras fases de un pipeline de Machine Learning clásico utilizando el conocido dataset de Iris. La meta es llevar a cabo la lectura, ingeniería de características, análisis visual y el preprocesamiento riguroso de los datos para dejarlos estructurados y escalados, listos para la futura fase de entrenamiento de modelos predictivos.



---

### 1. Recopilación y Carga Inicial
* Lectura del archivo `iris_data.csv`.
* Exploración inicial de la estructura del dataset y eliminación de la columna `Id`, al ser un identificador único que no aporta valor analítico ni predictivo.

### 2. Ingeniería de Características (Feature Engineering)
Se crearon nuevas métricas lógicas a partir de los datos existentes para ayudar a los futuros modelos a encontrar patrones más complejos:
* **Cálculo de Áreas:** Creación de las variables `SepalAreaCm2` y `PetalAreaCm2` multiplicando la longitud y anchura respectiva del sépalo y del pétalo.

### 3. Análisis Exploratorio de Datos (EDA) y Visualización
Generación de gráficos estadísticos para comprender la distribución y relación de las variables antes de modelar:
* 📊 **Relaciones Multivariables:** Un *pairplot* para visualizar de forma matricial cómo interactúan las características entre sí, segmentadas por la especie de la flor.
* 🌡️ **Mapa de Calor (Heatmap):** Una matriz de correlación para identificar colinealidad entre las variables numéricas originales y las nuevas características de área.
* 📦 **Detección de Outliers:** Un diagrama de caja (*boxplot*) para analizar la dispersión de los datos y detectar posibles valores atípicos en las mediciones florales.

### 4. Preprocesamiento de los Datos
Preparación final y estandarización del dataset para su consumo por algoritmos de Machine Learning:
* **Separación y Codificación:** Aislamiento de la variable objetivo (`Species`) y aplicación de *Label Encoding* para transformar las categorías de texto en valores numéricos (0, 1, 2).
* **División del Dataset:** Separación de los datos en conjuntos de entrenamiento (80%) y prueba (20%) utilizando partición estratificada para mantener la proporción de las clases.
* **Escalamiento (Feature Scaling):** Estandarización de las características utilizando `StandardScaler` (ajustado únicamente con los datos de entrenamiento para evitar fuga de información) con el fin de asegurar que todas las variables contribuyan equitativamente.

---

## 💻 Tecnologías Utilizadas
* **Python 3.x**
* **Pandas & NumPy:** Manipulación de datos y operaciones numéricas.
* **Matplotlib & Seaborn:** Visualización de datos y gráficos estadísticos.
* **Scikit-learn (sklearn):** Herramientas de preprocesamiento, codificación de etiquetas y división de datasets.
