"""
Análisis Inicial y Exploración de Datos: Dataset de Vuelos
Objetivo: Ejecutar las primeras etapas del ciclo de vida de los datos 
(Carga, Limpieza y Análisis Exploratorio) dejando el dataset listo para modelado.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuraciones visuales para los graficos y definición de la carpeta
# donde se van a guardar

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

output_dir = "graficas_output_clase2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta creada: {output_dir}")
else:
    print(f"Las gráficas se guardarán en la carpeta existente: {output_dir}")
# ETAPA 1: RECOPILACIÓN Y CARGA DE DATOS

print("Iniciando la carga de datos...")
df = pd.read_csv("Scraped_dataset.csv")

# Limpiamos los nombres de las columnas
df.columns = df.columns.str.strip().str.replace(" ", "_")

print(f"Dataset cargado exitosamente. Dimensiones iniciales: {df.shape}")

# ETAPA 2: LIMPIEZA DE DATOS Y INGENIERIA DE CARACTERISTICAS

print("Iniciando limpieza de datos y extracción de características...")

# Limpiamos la columna de precio, ya que viene con comas
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

# Separamos los datos de la columna Airline-Class,
# ya que trae la compañia y la clase en un mismo dato
def extract_airline_info(text):
    parts = str(text).split('\n')
    if len(parts) >= 3:
        return parts[0].strip(), parts[2].strip()
    return "Unknown", "Unknown"

df[['Airline', 'Class']] = df['Airline-Class'].apply(
    lambda x: pd.Series(extract_airline_info(x))
)

# Sacamos las ciudades de los campos de departure_time y arrival_time
# ya que viene junto con la hora de llegada y salida
df['Source_City'] = df['Departure_Time'].str.split('\n').str[1]
df['Destination_City'] = df['Arrival_Time'].str.split('\n').str[1]

# Estandarizamos la duración edl vuelo a minutos totales
def convert_duration(duration):
    duration = str(duration)
    h = 0
    m = 0
    if 'h' in duration:
        h = int(duration.split('h')[0])
        if 'm' in duration:
            m_str = duration.split('h')[1].replace('m', '').strip()
            m = int(m_str) if m_str.isdigit() else 0
    elif 'm' in duration:
        m = int(duration.replace('m', '').strip())
    return (h * 60) + m

df['Duration_Mins'] = df['Duration'].apply(convert_duration)

# Limpiamos el numero de escalas
def clean_stops(stops):
    stops = str(stops).lower()
    if "non-stop" in stops: return 0
    if "1-stop" in stops: return 1
    try:
        return int(stops.split('-')[0])
    except:
        return 0

df['Total_Stops_Clean'] = df['Total_Stops'].apply(clean_stops)

# Seleccionamos el dataset limpio y las columnas que necesitamos
features = [
    'Airline', 'Class', 'Source_City', 'Destination_City', 
    'Duration_Mins', 'Total_Stops_Clean', 'Price'
]
df_clean = df[features].copy()
df_clean.dropna(inplace=True) # borramos las lineas que tengan datos faltantes

print("Limpieza finalizada. Mostrando las primeras filas del dataset limpio:")
print(df_clean.head())

# ETAPA 3: ANÁLISIS EXPLORATORIO DE DATOS CON GRAFICAS

print("\nGenerando visualizaciones exploratorias...")

# Distribución General de los Precios
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Price'], bins=50, kde=True, color='blue')
plt.title('Distribución de los Precios de Vuelos')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
save_path = os.path.join(output_dir, 'distribucion_precios.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# Relación entre la Clase del Vuelo y el Precio
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x='Class', y='Price', palette='Set2', hue='Class')
plt.title('Distribución de Precios por Clase de Vuelo')
plt.xlabel('Clase')
plt.ylabel('Precio')
save_path = os.path.join(output_dir, 'precio_por_clase.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# Precio Promedio por Aerolínea
plt.figure(figsize=(12, 6))
sns.barplot(data=df_clean, x='Airline', y='Price', estimator=np.mean, errorbar=None, palette='viridis', hue='Class')
plt.title('Precio Promedio por Aerolínea')
plt.xlabel('Aerolínea')
plt.ylabel('Precio Promedio')
plt.xticks(rotation=45)
save_path = os.path.join(output_dir, 'precio_por_aerolinea.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

print("✓ Gráficos exploratorios guardados como imágenes (.png)")


"""
=======================================================================
Desde aqui podemos seguir con el resto de etapas como el preprocesamiento,
la selección del modelo, el entrenamiento de este, etc. En esta lecture solo
llegamos hasta el analisis de los datos y la ingenieria de características +
algunas gráficas exploratorias del dataset ya limpio.
=======================================================================
"""
