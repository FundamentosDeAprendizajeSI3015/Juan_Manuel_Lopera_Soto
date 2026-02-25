# Pipeline de Machine Learning: Clasificación con Dataset FIFA 23

> **Contexto:** Esta carpeta contiene la práctica correspondiente a la etapa de entrenamiento, evaluación e interpretabilidad de modelos basados en árboles para el curso universitario de **Fundamentos de Aprendizaje Automático**.

## 🎯 Objetivo de la Práctica

El objetivo principal de esta etapa es diseñar y ejecutar un pipeline de clasificación de Machine Learning utilizando el conjunto de datos de jugadores de FIFA 23. La meta es entrenar modelos basados en ensambles de árboles (Random Forest y Gradient Boosting) capaces de predecir la mejor posición de un futbolista (`Best Position`) basándose en sus atributos técnicos y físicos, aplicando buenas prácticas de preprocesamiento, optimización de hiperparámetros y prevención de fuga de datos (*data leakage*).

---

### 1. Recopilación y Carga Inicial
* Lectura del archivo `FIFA23Data.csv` que contiene registros detallados de miles de futbolistas.
* Separación de las variables predictoras ($X$) y la variable objetivo ($y$).

### 2. Preprocesamiento y Prevención de Fuga de Datos
Preparación del dataset garantizando que el modelo aprenda patrones genuinos y no memorice respuestas directas:
* **Eliminación de Fuga de Datos (*Data Leakage*):** Remoción estratégica de columnas de valoraciones por posición específica (ej. `ST Rating`, `RW Rating`, `GK Rating`) y variables descriptivas (nombres, URLs, IDs) que no aportan a la generalización del modelo.
* **Pipelines de Transformación:** Implementación de `ColumnTransformer` en Scikit-learn para procesar paralelamente distintos tipos de datos.
* **Codificación (Ordinal Encoding):** Transformación de variables categóricas de texto (como el pie preferido o las tasas de trabajo) a valores numéricos discretos compatibles con los algoritmos basados en árboles.

### 3. Entrenamiento y Optimización de Modelos
Construcción y ajuste fino de dos arquitecturas de ensamble:
* **Random Forest Classifier:** Entrenamiento de un bosque aleatorio robusto, aprovechando el procesamiento multihilo.
* **HistGradientBoosting Classifier:** Implementación de la versión optimizada basada en histogramas de Gradient Boosting, ideal para acelerar el entrenamiento en datasets tabulares grandes utilizando la CPU.
* **Búsqueda en Malla (GridSearchCV):** Exploración sistemática de hiperparámetros (`max_depth`, `min_samples_leaf`, `n_estimators`/`max_iter`) mediante validación cruzada ($k$-fold = 3) para encontrar la configuración que mitigue el sobreajuste (*overfitting*).

### 4. Evaluación del Rendimiento
Medición de la capacidad predictiva de los modelos sobre un conjunto de prueba aislado (20% de los datos):
* **Métricas Globales:** Cálculo de la precisión general (*Accuracy*) para comparar el rendimiento entre Random Forest y Gradient Boosting.
* **Reporte de Clasificación:** Análisis detallado de la precisión, exhaustividad (*recall*) y el *F1-score* por cada clase (posición), identificando los retos del modelo al diferenciar posiciones solapadas físicamente (ej. laterales vs. carrileros).

### 5. Interpretabilidad y Visualización
Generación automática de gráficos estáticos (almacenados en `graficas_output_lecture6/`) para entender cómo los algoritmos toman decisiones:

* 📊 **Matriz de Confusión:** Visualización de los aciertos y errores de clasificación por posición, revelando patrones de confusión entre roles similares en el campo.

* 📈 **Importancia de Atributos (Feature Importance):** Gráfico de barras horizontales (`barh`) que extrae del pipeline entrenado las 15 estadísticas (ej. *Defending Total*, *Finishing*) con mayor peso en la partición de los nodos.
* 🌳 **Estructura del Árbol:** Representación gráfica de un árbol de decisión individual extraído del bosque aleatorio (truncado a una profundidad de 3) para inspeccionar visualmente los umbrales lógicos y las reglas de partición.

---

## 💻 Tecnologías Utilizadas
* **Python 3.x**
* **Pandas & NumPy:** Manipulación de datos tabulares, filtrado de columnas y manejo de arreglos numéricos.
* **Matplotlib:** Configuración de estilos y renderizado de gráficos estadísticos y estructurales.
* **Scikit-learn (sklearn):** * `Pipeline` y `ColumnTransformer` para estructurar el flujo de datos.
  * `RandomForestClassifier` e `HistGradientBoostingClassifier` para el modelado.
  * `GridSearchCV` para la optimización paramétrica.
  * Herramientas de métricas (`accuracy_score`, `classification_report`, `ConfusionMatrixDisplay`) y extracción de árboles (`plot_tree`).