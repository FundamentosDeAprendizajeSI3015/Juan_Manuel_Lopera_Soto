# Pipeline de Machine Learning: Dataset Titanic

> **Contexto:** Esta carpeta contiene la práctica correspondiente a la etapa de procesamiento de datos para el curso universitario de **Fundamentos de Aprendizaje Automático**.

## 🎯 Objetivo de la Práctica

El objetivo principal de esta etapa es diseñar y ejecutar las primeras fases de un pipeline de Machine Learning clásico utilizando el conocido dataset del Titanic. La meta es llevar a cabo la lectura, análisis estadístico, detección visual de anomalías y el preprocesamiento riguroso de los datos para dejarlos limpios, codificados y escalados, listos para la futura fase de entrenamiento de modelos predictivos de supervivencia.

---

### 1. Recopilación y Carga Inicial
* Lectura del archivo `Titanic-Dataset.csv`.
* Exploración inicial de la estructura del dataset para identificar los tipos de datos y la cantidad de valores presentes en cada característica.

### 2. Cálculo de Medidas Estadísticas
Cálculo de métricas descriptivas para comprender la naturaleza de los datos antes de transformarlos:
* **Evaluación General:** Extracción de medidas de tendencia central (media, mediana), dispersión (desviación estándar) y posición (cuartiles) para las variables numéricas del conjunto de datos.

### 3. Análisis Exploratorio de Datos (EDA) y Visualización
Generación de gráficos estadísticos, guardados localmente, para comprender la distribución y detectar anomalías:
* 📊 **Distribución de Variables:** Histogramas para observar el comportamiento y los sesgos en variables continuas críticas, como la Edad (`Age`) y la Tarifa (`Fare`).
* 📦 **Detección de Outliers:** Gráficos de dispersión (*scatter plots*) y diagramas de caja (*boxplots*) orientados a identificar valores atípicos extremos que puedan afectar el rendimiento del modelo.
* 🌡️ **Mapa de Calor (Heatmap):** Una matriz de correlación para evaluar la colinealidad entre las características y entender qué variables tienen mayor relación con la supervivencia.

### 4. Preprocesamiento de los Datos
Preparación final, limpieza y estandarización del dataset para su consumo por algoritmos de Machine Learning:
* **Tratamiento de Nulos (Imputación):** Relleno inteligente de valores faltantes, utilizando la mediana para la característica `Age` (minimizando el impacto de outliers) y la moda para el puerto de embarque (`Embarked`).
* **Limpieza de Variables:** Eliminación de columnas irrelevantes que funcionan como identificadores (`PassengerId`, `Name`, `Ticket`) o que poseen una cantidad excesiva de valores nulos inmanejables (`Cabin`).
* **Codificación (Label Encoding):** Transformación de las variables categóricas de texto (`Sex`, `Embarked`) a valores numéricos comprensibles por los modelos.
* **Escalamiento (Feature Scaling):** Estandarización de las características numéricas (como `Age`, `Fare`, `Pclass`, `SibSp`, `Parch`) utilizando `StandardScaler` para asegurar que todas las variables tengan media 0 y varianza 1, contribuyendo equitativamente en cálculos de distancia.

### 5. Generación del Dataset Listo
* Exportación final de los datos limpios y transformados al archivo `Titanic-Dataset-Ready.csv`, preparado directamente para la ingesta en la fase de modelado.

---

## 💻 Tecnologías Utilizadas
* **Python 3.x**
* **Pandas & NumPy:** Manipulación de datos, imputaciones y operaciones numéricas.
* **Matplotlib & Seaborn:** Visualización de datos, análisis de distribuciones y gráficos estadísticos.
* **Scikit-learn (sklearn):** Herramientas de preprocesamiento, codificación de etiquetas (Label Encoding) y escalamiento de características (Standard Scaler).
