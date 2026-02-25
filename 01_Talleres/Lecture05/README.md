# Modelos Predictivos: Valor de Jugadores (FIFA 23) y Detección Cardíaca

> **Contexto:** Esta carpeta contiene la práctica correspondiente a la etapa de entrenamiento y evaluación de modelos de regresión lineal regularizada y regresión logística para la materia de **Aprendizaje Automático**. Se abordan dos problemas fundamentales: la predicción de valores continuos y la clasificación binaria médica.

## 🎯 Objetivos de la Práctica

El objetivo principal de esta etapa es diseñar flujos de trabajo (pipelines) de Machine Learning para resolver dos problemáticas distintas superando los desafíos de dimensionalidad, escala y costo computacional:
1. **Regresión Lineal (FIFA 23):** Predecir el valor de mercado de un jugador de fútbol (en millones de euros) basándose en sus características de rendimiento deportivo aplicando técnicas de regularización (Ridge y LASSO).
2. **Regresión Logística (Cardiología):** Determinar la presencia o ausencia de una enfermedad cardíaca en pacientes a partir de datos médicos clínicos, optimizando el umbral de decisión y evaluando el rendimiento mediante métricas de clasificación.

---

### 1. Recopilación y Carga Inicial
* **Dataset FIFA 23 (`FIFA23Data.csv`):** Selección de características numéricas clave para el rendimiento deportivo (`Age`, `Overall`, `Potential`, `Pace Total`, `Shooting Total`, etc.) y definición de la variable objetivo continua (`Value(in Euro)`).
* **Dataset Médico (`data.csv`):** Carga de datos clínicos (edad, colesterol, presión arterial, etc.). Se realizó una limpieza profunda reemplazando caracteres nulos (`?`) por `NaN` y eliminando registros incompletos.
* **Binarización del Target:** Para el caso médico, la variable predictiva continua (`num`) se transformó en un formato binario estructurado (0 = Sano, >0 = Enfermo) para habilitar la clasificación algorítmica.

### 2. Preprocesamiento y Transformación de Datos
Preparación de los datos para su consumo por los algoritmos, mitigando problemas de escala y explosión de memoria:
* **Escalamiento del Target (Regresión):** División de la variable objetivo de FIFA entre 1,000,000 para realizar predicciones en "Millones de Euros", previniendo el desbordamiento matemático.
* **Transformación Polinomial (`PolynomialFeatures`):** Expansión del espacio de características (grados 1 a 3 en ambos datasets) para capturar las relaciones no lineales entre las variables independientes y su respectivo objetivo.
* **Estandarización (`StandardScaler`):** Normalización de las características para asegurar que todas las variables tengan media 0 y varianza 1, un paso estricto y necesario para el correcto funcionamiento de la regularización y la convergencia logística.

### 3. Optimización de Hiperparámetros y Entrenamiento
Uso de validación cruzada para encontrar la mejor configuración de los modelos de forma automatizada:
* **Definición de Pipelines:** Construcción de flujos de trabajo integrados para aplicar la transformación polinomial, el escalado y el estimador final en un solo paso.
* **Búsqueda Aleatoria (`RandomizedSearchCV`):** Exploración de distribuciones de parámetros (como el grado del polinomio, la penalización `alpha` para lineal o el inverso de regularización `C` para logística) usando una distribución recíproca con validación cruzada (`cv=4`) y paralelización (`n_jobs=-1`).
* **Entrenamiento de Modelos:** Ajuste de tres enfoques distintos según el problema:
  * **Ridge Regression (L2):** Penaliza el tamaño de los coeficientes para evitar el sobreajuste, logrando alta precisión en predicción de valor.
  * **LASSO Regression (L1):** Fomenta la escasez llevando coeficientes irrelevantes a cero.
  * **Logistic Regression:** Clasificador probabilístico configurado con el optimizador avanzado `saga` y un alto límite de iteraciones para asegurar la convergencia en el espacio polinomial multidimensional médico.

### 4. Evaluación de Modelos y Visualización de Resultados
Cálculo de métricas de error/precisión y generación de gráficos estadísticos para interpretar el rendimiento:
* **Métricas de Regresión (FIFA 23):** Extracción del Coeficiente de Determinación ($R^2$) y el Error Absoluto Medio (MAE) para cuantificar la varianza explicada y el margen de error promedio en millones de euros.
* **Métricas de Clasificación (Detección Cardíaca):** Evaluación exhaustiva utilizando la Precisión Global (`Accuracy`) y el valor `F1-score` para balancear falsos positivos y falsos negativos en el diagnóstico médico.
* 📊 **Gráficas y Reportes Visuales:** * Generación de gráficos de dispersión exportados de forma automática al directorio `graficas_output_lecture5_lineal` para comparar predicciones vs. valores reales de mercado.
  * Trazado e interpretación de la **Matriz de Confusión** mediante `ConfusionMatrixDisplay` para visualizar la proporción de pacientes correctamente diagnosticados frente a los errores del modelo médico.

---

## 💻 Tecnologías Utilizadas
* **Python 3.x**
* **Pandas & NumPy:** Carga de datos, filtrado, manejo de valores nulos y manipulación de arreglos matriciales.
* **Scikit-learn (sklearn):** * Construcción del flujo de trabajo (`Pipeline`).
  * Preprocesamiento de variables (`StandardScaler`, `PolynomialFeatures`).
  * Modelos predictivos (`Ridge`, `Lasso`, `LogisticRegression`).
  * Optimización de hiperparámetros (`RandomizedSearchCV`, `train_test_split`).
  * Métricas de validación (`r2_score`, `mean_absolute_error`, `accuracy_score`, `f1_score`, `confusion_matrix`, `ConfusionMatrixDisplay`).
* **SciPy (`scipy.stats`):** Generación de distribuciones estadísticas continuas (`reciprocal`) para la búsqueda hiperparamétrica.
* **Matplotlib:** Creación, formateo estructurado y guardado de las gráficas de evaluación (dispersión y matrices de confusión).