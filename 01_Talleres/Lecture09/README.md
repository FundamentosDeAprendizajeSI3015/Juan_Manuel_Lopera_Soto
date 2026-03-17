# Pipeline de Clustering: Análisis No Supervisado de Datos (FIRE UdeA)

Este proyecto implementa un ecosistema de Machine Learning no supervisado diseñado para el descubrimiento de patrones y segmentación de datos en el contexto de FIRE UdeA. El repositorio contiene implementaciones adaptables que procesan tanto datasets realistas (con valores nulos y variables categóricas) como datasets estrictamente numéricos, garantizando **escalabilidad, manejo dinámico de variables y visualización efectiva**.

## 🎯 Objetivo General
El objetivo principal es aplicar algoritmos de clustering para identificar agrupaciones naturales dentro de los datos financieros y operativos sin depender de etiquetas previas. Esto permite descubrir perfiles subyacentes, detectar anomalías (ruido) y segmentar las observaciones basándose en su similitud multidimensional.

## 🛠️ Metodología y Arquitectura
El pipeline se fundamenta en la combinación de dos enfoques de agrupamiento complementarios: **K-Means** (basado en centroides y particiones euclidianas) y **DBSCAN** (basado en densidad espacial).

### 🏗️ Ingeniería de Datos (Pipeline)
Para garantizar el correcto cálculo de distancias y la estabilidad de los algoritmos, se utiliza un `ColumnTransformer` que automatiza el preprocesamiento dinámico según la naturaleza del dataset:

* **Imputación:** Uso dinámico de `SimpleImputer` con la estrategia de mediana para variables numéricas y el valor más frecuente para variables categóricas, asegurando la ejecución sin fallos ante datos faltantes.
* **Escalamiento:** Aplicación de `StandardScaler` en todas las variables numéricas. Esto es crítico en algoritmos basados en distancias para evitar que variables con magnitudes altas (ej. ingresos totales) dominen sobre tasas o ratios pequeños.
* **Codificación:** Transformación de variables de texto mediante `OneHotEncoder` (como la columna 'unidad'), permitiendo su inclusión en el espacio matemático.
* **Reducción de Dimensionalidad:** Integración de **PCA** (*Principal Component Analysis*) para proyectar el espacio multidimensional a 2 componentes principales (2D), habilitando la visualización de los clústeres.

---

## 📊 Análisis de Resultados y Visualizaciones
El código genera automáticamente una batería de 5 gráficas que se exportan a un directorio local (`graficas_clustering` o `graficas_clustering_udea`), permitiendo auditar el comportamiento espacial de los datos:

1.  **01. Visualización de Datos (PCA 2D):** Proyecta la distribución original de los datos preprocesados en un plano bidimensional para observar su topología antes del agrupamiento.
2.  **02. K-Means (K=2):** Muestra una partición binaria del espacio, forzando al algoritmo a encontrar las dos tendencias más marcadas del dataset.
3.  **03. Curva de Inercia (Método del Codo):** Gráfica fundamental para evaluar el rango de $K$ (de 1 a 10). Permite identificar el punto de inflexión donde añadir más clústeres deja de aportar un beneficio significativo en la reducción de la varianza interna.
4.  **04. K-Means (K=4):** Representación del agrupamiento ajustado a 4 centroides, permitiendo observar una segmentación más granular.
5.  **05. DBSCAN:** Visualiza agrupaciones basadas en la densidad de los puntos. Es especialmente útil para mapear geometrías no esféricas y separar elementos atípicos (*outliers*), los cuales son clasificados como ruido (etiqueta -1).

---

## 📈 Métricas de Evaluación Robustas
Dado que el aprendizaje es no supervisado, el rendimiento no se evalúa con precisión frente a una etiqueta, sino mediante la cohesión y separación de los clústeres:

* **Inercia (WCSS - Within-Cluster Sum of Squares):** Mide la compacidad de los clústeres en K-Means. Representa la suma de las distancias al cuadrado de cada punto a su centroide asignado. El objetivo es minimizar este valor sin caer en el sobreajuste de tener tantos clústeres como puntos:
    $$WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2$$
* **Frecuencia de Ruido (DBSCAN):** Evaluación de la cantidad de puntos etiquetados como ruido frente a los puntos centralizados, útil para detectar anomalías en los registros operativos o financieros.

## 🚀 Requisitos
Para ejecutar este pipeline, asegúrate de tener el siguiente entorno configurado:

```bash
pip install pandas numpy matplotlib scikit-learn