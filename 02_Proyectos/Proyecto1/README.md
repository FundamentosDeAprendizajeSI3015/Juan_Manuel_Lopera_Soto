# 🧠 Doomscrolling Detector: Inferencia de Estado Emocional mediante Telemetría

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🎓 Información Académica
* **Institución:** Universidad EAFIT
* **Curso:** Fundamentos de Aprendizaje Automático
* **Estudiante:** Juan Manuel Lopera Soto
* **Docente:** Olga Lucía Quintero Montoya

---

## 🎯 Descripción del Proyecto

Este proyecto aborda el fenómeno del **"Doomscrolling"** (consumo compulsivo de contenido digital) aplicando el ciclo de vida completo del Machine Learning. 

El objetivo principal es predecir el impacto emocional en tiempo real de un usuario basándonos **exclusivamente en telemetría de comportamiento cinético** (velocidad de *scroll*, tiempo de uso, plataforma, interacciones), evitando depender de diagnósticos médicos previos o encuestas subjetivas para prevenir el *Data Leakage* (Fuga de Información).

## 🗂️ El Dataset

Se generó un dataset sintético de **2,500 registros** fundamentado en lógica de negocio y psicología del comportamiento digital:
* **Variables Predictoras (X):** `Daily_Usage_Time_min`, `Scroll_Rate_ppm`, `Platform`, `Likes`, `Comments`, demografía.
* **Variable Objetivo Original (y):** `Emotional_State_Post_Usage` (Ansioso, Feliz, Deprimido, Neutro, etc.).
* **Reglas de Negocio:** Se integró un "Índice de Doomscrolling" oculto donde el alto consumo de tiempo combinado con un *Scroll Rate* acelerado degrada el índice de salud mental del usuario.

---

## ⚙️ Metodología y Pipeline de Machine Learning

El proyecto se estructuró en 4 fases técnicas fundamentales:

### 1. Ingeniería de Datos (Data Prep)
* **Limpieza:** Imputación de valores nulos (media para numéricos, moda para categóricos) y filtrado de anomalías lógicas.
* **Transformaciones:** Aplicación de Log-Transform para corregir sesgos en el tiempo de uso.
* **Encoding & Scaling:** One-Hot Encoding para plataformas, Label Encoding para el target y normalización mediante `StandardScaler` (crucial para los algoritmos basados en distancias).

### 2. Aprendizaje No Supervisado (Identificación de Clusters)
Se le ocultaron las etiquetas originales a la IA para evaluar si el comportamiento formaba grupos naturales.
* **K-Means:** Logró segmentar a los usuarios en perfiles de comportamiento compacto.
* **DBSCAN:** Logró segmentar a los usuarios en clases mas diversas poco diferenciables para el resto de los modelos, los cuales a diferencia de DBSCAN tomaron un camino mas preciso diviendo el dataset en solo 3 clases, DBSCAN logro identificar clases menores que eran muy similares a las principales.
* **Fuzzy C-Means:** Permitió evaluar probabilidades de pertenencia difusa, ideal para el espectro de la psicología humana.
* **Clustering Jerárquico:** Generación de un **Dendrograma** para visualizar las distancias topológicas entre comportamientos sanos y compulsivos.

### 3. Aprendizaje Supervisado (Modelos Predictivos)
Se reintrodujeron las etiquetas para entrenar modelos de clasificación multiclase.
* **Regresión Logística Multinomial:** Utilizada como modelo *Baseline* (Línea base).
* **Árboles de Decisión:** Elegidos por su **interpretabilidad clínica** (modelos de caja blanca). 
* **Feature Importance:** El análisis del árbol reveló que el `Scroll_Rate_ppm` es el principal predictor del estado emocional, superando a la plataforma en sí.

### 4. El Experimento Final (Re-evaluación de Etiquetas)
¿Qué es más predecible: las emociones reportadas por humanos o los grupos matemáticos encontrados por K-Means?
* Se entrenaron modelos supervisados idénticos para predecir: a) La emoción original, b) El clúster de K-Means, c) El clúster Jerárquico.
* **Conclusión:** El modelo predictivo alcanzó un *F1-Score* significativamente mayor al predecir las etiquetas generadas por K-Means/Jerárquico. Esto prueba que el comportamiento cinético forma agrupaciones mucho más puras y predecibles que las emociones auto-reportadas (que suelen contener ruido por la subjetividad humana).

---

## 🚀 Estructura del Repositorio y Ejecución

El código está modularizado para evaluar cada fase del pipeline de forma independiente:

```text
├── main.py                  # Carga, limpieza, EDA, ingeniería de características y exportación
├── unsupervised.py          # K-Means, DBSCAN y Análisis de Componentes Principales (PCA)
├── unsupervised_2.py        # Fuzzy C-Means (skfuzzy) y Clustering Jerárquico (Dendrogramas)
├── supervised.py            # Regresión Logística y Árboles de Decisión (Feature Importance)
├── comparativa_final.py     # Experimento cruzado de métricas (Human labels vs Machine clusters)
├── graficas_exportadas/     # Carpeta autogenerada con matrices de confusión, dendrogramas y EDA
└── social_media_addiction_2500.csv # Dataset principal