# Detector de Doomscrolling Basado en Datos de Telemetria

## Descripción del Proyecto

Este proyecto aborda una problemática crítica en la era digital: el impacto silencioso de las redes sociales en la salud mental. En lugar de predecir métricas de negocio (clicks o tiempo de retención), este sistema actúa como un **Motor de Inferencia de Bienestar Digital**.

El objetivo es desarrollar un modelo de Machine Learning capaz de predecir el **Estado Emocional Post-Uso** (`Emotional_State_Post_Usage`) de un usuario, basándose exclusivamente en métricas de comportamiento cinético y patrones de consumo (scroll, tiempo, interacciones) en tiempo real, sin depender de encuestas ni diagnósticos médicos previos.

## Descripción del Dataset

El conjunto de datos contiene 2,500 registros de comportamiento de usuarios en diversas plataformas (TikTok, Instagram, LinkedIn, etc.).

### Variables de Entrada (Features)
Estas variables representan la "huella digital" del comportamiento:
* `Daily_Usage_Time_min`: Tiempo total de uso (minutos).
* `Scroll_Rate_ppm`: Velocidad de desplazamiento (píxeles por minuto). Indicador clave de ansiedad/compulsión.
* `Interactions`: Likes, Comentarios, Posts, Mensajes enviados.
* `Platform`: Red social utilizada.
* `Demographics`: Edad y Género.

### Variable Objetivo (Target)
* `Emotional_State_Post_Usage`: Variable categórica multiclase (ej: *Happy, Neutral, Anxious, Sad, Angry*).

### Prevención de Data Leakage
Para garantizar la validez del modelo en un entorno productivo real, se eliminaron deliberadamente las siguientes columnas durante el pre-procesamiento, ya que constituyen "fugas de información" (información que no tendríamos en tiempo real o que es consecuencia directa del target):
* `Addiction_Level` (Diagnóstico realizado posteriormente, no es en tiempo real).
* `Mental_Health_Index` (Puntaje clínico).
* `Productivity_Loss_Score`.

## Ingeniería de Datos y Pipeline

El proyecto implementa un pipeline robusto de pre-procesamiento:

1.  **Limpieza de Datos:**
    * Imputación de valores nulos (Media para numéricos, Moda para categóricos).
    * Filtrado lógico (eliminación de tiempos de uso > 24h).
2.  **Transformaciones Matemáticas:**
    * **Log Transformation:** Aplicada a `Daily_Usage_Time_min` para corregir el sesgo positivo (cola larga) y normalizar la distribución de usuarios extremos ("Power Users").
3.  **Encoding:**
    * **One-Hot Encoding:** Para la variable `Platform`, permitiendo al modelo tratar cada red social equitativamente sin imponer un orden numérico falso.
    * **Label Encoding:** Para la variable objetivo.
4.  **Escalado:**
    * Uso de `StandardScaler` para normalizar características numéricas (`Age`, `Scroll_Rate`), crucial para algoritmos basados en distancias o gradientes.

## Visualización y Análisis Exploratorio (EDA)

Durante la exploración se identificaron patrones clave:
* **El "Efecto TikTok":** Se observó una correlación directa entre plataformas de video corto y un `Scroll_Rate_ppm` elevado.
* **Distribución de Uso:** La transformación logarítmica fue necesaria para estabilizar la varianza en los datos de tiempo de uso.

## Modelado (Estrategia Propuesta)

El proyecto está diseñado para evaluar y comparar los siguientes enfoques:
* **Baseline:** Regresión Logística Multinomial.
* **Modelo Avanzado:** Gradient Boosting (CatBoost o LightGBM), elegidos por su capacidad superior para manejar variables categóricas y relaciones no lineales en datos tabulares.

**Métrica de Evaluación:** `F1-Score (Macro)`, priorizando el equilibrio entre Precision y Recall para no ignorar las clases minoritarias de riesgo (ej: Depresión severa).

## Requisitos e Instalación

1.  Clonar el repositorio:
    ```bash
    git clone [https://github.com/tu-usuario/doomscrolling-detector.git](https://github.com/tu-usuario/doomscrolling-detector.git)
    ```
2.  Instalar dependencias:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```
3.  Ejecutar el pipeline de limpieza:
    ```bash
    python src/data_pipeline.py
    ```

---
*Proyecto desarrollado para el curso de Fundamentos de Aprendizaje Automático.*
