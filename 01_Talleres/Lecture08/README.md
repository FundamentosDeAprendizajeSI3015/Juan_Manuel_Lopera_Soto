# Pipeline Integral de Clasificación: Detección de Estados de Tensión (FIRE UdeA)

Este proyecto implementa un ecosistema completo de Machine Learning diseñado para la clasificación binaria de estados de salud (**Sano** vs. **Tensión**), utilizando un dataset realista basado en el contexto de FIRE UdeA. El modelo no solo busca precisión, sino también **interpretabilidad clínica y robustez estadística**.

## 🎯 Objetivo General
El objetivo principal es desarrollar un modelo predictivo capaz de identificar patrones de riesgo (Tensión) a partir de variables multimodales, asegurando que el modelo generalice correctamente en datos no vistos y que sus decisiones sean explicables para un humano.

## 🛠️ Metodología y Arquitectura
El modelo se basa en el algoritmo **Gradient Boosting Classifier**, configurado bajo una estrategia de "aprendizaje conservador" para mitigar el sobreajuste (*overfitting*) en datasets de tamaño pequeño o moderado.

### 🏗️ Ingeniería de Datos (Pipeline)
Para garantizar la integridad de los datos y evitar el *data leakage*, se utiliza un `ColumnTransformer` que automatiza:
* **Imputación:** Uso de la mediana para variables numéricas y el valor más frecuente para categóricas (robusto a valores atípicos).
* **Escalamiento:** `StandardScaler` para normalizar las magnitudes de las variables numéricas.
* **Codificación:** `OneHotEncoder` para transformar variables categóricas (como la 'unidad') en formato procesable por el modelo.
* **Validación:** División estricta del dataset en **70% Entrenamiento**, **15% Validación** y **15% Prueba**.



## 📊 Análisis de Resultados y Visualizaciones
El pipeline genera automáticamente 8 gráficas maestras que permiten auditar el modelo desde diferentes ángulos:

1.  **01. Proyección UMAP:** Reduce la dimensionalidad para visualizar si los casos de "Tensión" y "Sano" forman clústeres naturales en un espacio 2D.
2.  **02. Matriz de Confusión:** Muestra el conteo de Verdaderos Positivos, Falsos Positivos, etc., para evaluar el sesgo del modelo.
3.  **03. Curva ROC:** Evalúa el compromiso entre sensibilidad y especificidad. Un área bajo la curva (AUC) cercana a 1.0 indica una excelente separación de clases.
4.  **04. Curva Precision-Recall:** Crucial para datasets donde la prevalencia de la enfermedad es baja, enfocándose en la calidad de las predicciones positivas.
5.  **05. Curva de Calibración:** Comprueba si las probabilidades predichas (ej. 0.8) realmente corresponden a la frecuencia real de los eventos (80% de los casos).
6.  **06. Importancia de Variables:** Gráfico de barras que identifica qué sensores o métricas tienen mayor peso en las decisiones del modelo.
7.  **07. Análisis SHAP:** Utiliza teoría de juegos para explicar el impacto positivo o negativo de cada variable en una predicción individual.
    
8.  **08. Árbol de Decisión:** Visualiza las reglas lógicas (IF-THEN) del primer estimador del modelo para entender su razonamiento inicial.

## 📈 Métricas de Evaluación Robustas
Más allá del Accuracy, este modelo se evalúa mediante:
* **Log Loss:** Mide la incertidumbre de las predicciones.
* **Brier Score:** Evalúa la precisión de las probabilidades asignadas.
    $$BS = \frac{1}{N} \sum_{t=1}^{N} (f_t - o_t)^2$$
* **F1-Score:** El balance armónico entre Precisión y Recall.

## 🚀 Requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap umap-learn