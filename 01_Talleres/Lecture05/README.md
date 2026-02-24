# Predicción de Valor de Jugadores: Dataset FIFA 23

> **Contexto:** Esta carpeta contiene la práctica correspondiente a la etapa de entrenamiento y evaluación de modelos de regresión lineal regularizada para la materia de **Aprendizaje Automático**.

## 🎯 Objetivo de la Práctica

El objetivo principal de esta etapa es diseñar un pipeline de Machine Learning para predecir el valor de mercado de un jugador de fútbol (en millones de euros) basándose en sus características numéricas (edad, velocidad, tiro, pase, etc.). La meta es aplicar técnicas de regularización (Ridge y LASSO), realizar transformaciones polinomiales, optimizar hiperparámetros mediante búsqueda aleatoria y evaluar el rendimiento del modelo superando los desafíos de escala y costo computacional.



---

### 1. Recopilación y Carga Inicial
* Lectura del archivo `FIFA23Data.csv`.
* Selección de características numéricas clave para el rendimiento deportivo (`Age`, `Overall`, `Potential`, `Pace Total`, `Shooting Total`, `Passing Total`, `Dribbling Total`, `Defending Total`, `Physicality Total`) y definición de la variable objetivo (`Value(in Euro)`).
* Filtrado y limpieza de registros con valores nulos en las características seleccionadas para asegurar la estabilidad durante el entrenamiento.

### 2. Preprocesamiento y Transformación de Datos
Preparación de los datos para su consumo por los algoritmos de regresión, mitigando problemas de escala y explosión de memoria:
* **Escalamiento del Target:** División de la variable objetivo entre 1,000,000 para realizar predicciones en "Millones de Euros", previniendo el desbordamiento matemático y facilitando la convergencia del modelo.
* **Transformación Polinomial (`PolynomialFeatures`):** Expansión del espacio de características (grados 1 a 3) para capturar las relaciones no lineales entre las estadísticas del jugador y su valor de mercado.
* **Estandarización (`StandardScaler`):** Normalización de las características para asegurar que todas las variables tengan media 0 y varianza 1, un paso estricto y necesario para el correcto funcionamiento de la regularización.

### 3. Optimización de Hiperparámetros y Entrenamiento

Uso de validación cruzada para encontrar la mejor configuración de los modelos de forma automatizada:
* **Definición de Pipelines:** Construcción de flujos de trabajo integrados para aplicar la transformación polinomial, el escalado y el regresor final en un solo paso.
* **Búsqueda Aleatoria (`RandomizedSearchCV`):** Exploración de distribuciones de parámetros (como el grado del polinomio y el parámetro de penalización `alpha` usando una distribución recíproca) con validación cruzada de 4 pliegues (`cv=4`).
* **Entrenamiento de Modelos Regularizados:** Ajuste de dos modelos distintos para comparar su comportamiento:
  * **Ridge Regression (L2):** Penaliza el tamaño de los coeficientes para evitar el sobreajuste, resultando en un modelo altamente preciso ($R^2$ superior al 90%).
  * **LASSO Regression (L1):** Fomenta la escasez (esparcimiento) llevando coeficientes irrelevantes a cero. Se ajustaron las iteraciones máximas y la tolerancia para asistir en su convergencia computacional.

### 4. Evaluación de Modelos y Visualización de Resultados
Cálculo de métricas de error y generación de gráficos estadísticos (guardados localmente) para interpretar el rendimiento:
* **Métricas de Rendimiento:** Extracción del Coeficiente de Determinación ($R^2$) y el Error Absoluto Medio (MAE) para cuantificar la varianza explicada y el margen de error promedio en millones de euros.
* 📊 **Gráficas de Dispersión (Real vs. Predicho):** Generación de gráficos exportados de forma automática al directorio local `graficas_output_lecture5_lineal`. Estas gráficas comparan las predicciones del modelo frente al valor real de prueba junto con una línea de "modelo ideal" para observar la distribución del error.

---

## 💻 Tecnologías Utilizadas
* **Python 3.x**
* **Pandas & NumPy:** Carga de datos, filtrado y manipulación de arreglos numéricos.
* **Scikit-learn (sklearn):** * Construcción de `Pipeline`.
  * Preprocesamiento (`StandardScaler`, `PolynomialFeatures`).
  * Modelos predictivos (`Ridge`, `Lasso`).
  * Optimización y métricas (`RandomizedSearchCV`, `train_test_split`, `r2_score`, `mean_absolute_error`).
* **SciPy (`scipy.stats`):** Generación de distribuciones estadísticas (`reciprocal`) para la búsqueda de hiperparámetros.
* **Matplotlib:** Creación, formateo y guardado local de las gráficas de evaluación de los modelos.
