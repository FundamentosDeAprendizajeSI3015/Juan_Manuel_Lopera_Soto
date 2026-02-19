# Análisis Exploratorio y Feature Engineering: Predicción de Vuelos

> **Contexto:** Esta carpeta contiene la práctica correspondiente a la Lecture/Semana 2 del curso de **Fundamentos de Aprendizaje Automático**.

## 🎯 Objetivo de la Práctica

El objetivo principal de esta etapa del proyecto es ejecutar las primeras fases del ciclo de vida de los datos aplicadas a un dataset desordenado sacado de kaggle. Partiendo de un conjunto de datos crudo y desestructurado, la meta es realizar la carga, limpieza profunda (Feature Engineering) y un Análisis Exploratorio de Datos (EDA) para dejar la información lista y depurada para futuros modelos de Machine Learning.

---

### 1. Recopilación y Carga Inicial
* Lectura del archivo `Scraped_dataset.csv`.
* Estandarización de los nombres de las columnas para evitar errores de sintaxis en el entorno de desarrollo (eliminación de espacios y saltos de línea).

### 2. Limpieza de Datos e Ingeniería de Características (Feature Engineering)
Al ser datos de vuelos y aerolineas desorganizados, se encontraron muchas columnas que contenian dos o mas datos dentro de esta, los cuales debiamos separar
* **Desglose de Aerolínea y Clase:** Separación de la columna `Airline-Class` (ej. *"SpiceJet \nSG-8169\nECONOMY"*) en dos columnas categóricas independientes: `Airline` y `Class`.
* **Procesamiento de Precios:** Transformación de la variable `Price` de formato texto con comas (ej. *"5,335"*) a valores numéricos continuos (`float`).
* **Transformación Temporal:** Conversión de la duración del vuelo (ej. *"02h 05m"*) a una variable numérica unificada: `Duration_Mins` (minutos totales).
* **Limpieza de Escalas:** Traducción de texto como *"non-stop"* o *"1-stop"* a variables numéricas discretas (0, 1, 2...).
* **Separación Geográfica:** Extracción de las ciudades de origen y destino a partir de las cadenas de fecha y hora.

### 3. Análisis Exploratorio de Datos (EDA) y Visualización
Luego de depurar los datos, generamos algunas graficas exploratorias para visualizar los cambios realizados al dataset:
* 📊 **Distribución de los Precios:** Un histograma para entender la concentración de vuelos económicos vs. costosos.
* 📦 **Precios por Clase de Vuelo:** Un diagrama de caja (*boxplot*) que evidencia la diferencia estadística entre las distintas clases (Economy vs. Business).
* 📈 **Precio Promedio por Aerolínea:** Un gráfico de barras para identificar qué aerolíneas dominan los segmentos de bajo costo y premium.

---

## 💻 Tecnologías Utilizadas
* **Python 3.x**
* **Pandas & NumPy:** Manipulación y limpieza de datos.
* **Matplotlib & Seaborn:** Visualización de datos.
