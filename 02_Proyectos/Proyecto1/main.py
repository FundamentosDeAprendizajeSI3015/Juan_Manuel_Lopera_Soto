import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# ESTILOS PARA LAS GRAFICAS

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# DIRECTORIO PARA LAS GRAFICAS
carpeta_graficas = "graficas_exportadas"
os.makedirs(carpeta_graficas, exist_ok=True)
print(f"Las gráficas se guardarán en el directorio: '{carpeta_graficas}'")

# CARGA DE DATOS

try:
    df = pd.read_csv('../../data/social_media_addiction_2500.csv')
    print("Dataset cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'social_media_addiction.csv' no se encuentra.")

# Miramos el estado inicial del dataset
print("\n--- INSPECCIÓN DEL DATASET ---")
print(df.head())
print(df.info())

# Borramos cualquier linea duplicada
duplicados = df.duplicated().sum()
print(f"\nFilas duplicadas eliminadas: {duplicados}")
df = df.drop_duplicates()

# Manejamos los valores nulos
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Limpiamos todas las categorias de texto
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# borramos la columna de ID, ya que no sirve para hacer predicciones
if 'User_ID' in df.columns:
    df = df.drop(columns=['User_ID'])

# hacemos una validacion logica para borrar tiempos imposibles
# o edades falsas (mayor a 100)
df = df[(df['Daily_Usage_Time_min'] <= 1440) & (df['Age'] <= 100)]


# ANALISIS DE LOS DATOS YA DEPURADOS

# Agregación y Agrupamiento
print("\n--- USO PROMEDIO POR PLATAFORMA ---")
print(df.groupby('Platform')['Daily_Usage_Time_min'].mean().sort_values(ascending=False))

# Medidas de Tendencia Central y Dispersión
col_analisis = 'Daily_Usage_Time_min'
print(f"\n--- ESTADÍSTICAS DE {col_analisis.upper()} ---")
print(f"Media: {df[col_analisis].mean():.2f}")
print(f"Mediana: {df[col_analisis].median():.2f}")
print(f"Desviación Std: {df[col_analisis].std():.2f}")
print(f"Mínimo: {df[col_analisis].min()} | Máximo: {df[col_analisis].max()}")

# Detección de Outliers por medio de IQR
Q1 = df[col_analisis].quantile(0.25)
Q3 = df[col_analisis].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR
outliers = df[df[col_analisis] > limite_superior]
print(f"Número de outliers detectados en Tiempo de Uso: {len(outliers)}")

# Histogramas
plt.figure(figsize=(10, 5))
sns.histplot(df['Daily_Usage_Time_min'], kde=True, bins=30, color='skyblue')
plt.title('Distribución del Tiempo de Uso Diario (Antes de Log Transform)')
plt.xlabel('Minutos')
plt.ylabel('Frecuencia')
plt.savefig(os.path.join(carpeta_graficas, '1_distribucion_uso_diario.png')) # <-- Guardado
plt.close() # <-- Cierre de figura

plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], kde=True, bins=20, color='salmon')
plt.title('Distribución de la Edad de los Usuarios')
plt.xlabel('Edad')
plt.savefig(os.path.join(carpeta_graficas, '2_distribucion_edad.png')) # <-- Guardado
plt.close() # <-- Cierre de figura

# Gráficos de Dispersión
# Relación entre Scroll Rate y Tiempo de Uso, coloreado por Plataforma
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Scroll_Rate_ppm', y='Daily_Usage_Time_min', hue='Platform', alpha=0.6)
plt.title('Relación: Velocidad de Scroll vs. Tiempo de Uso')
plt.xlabel('Scroll Rate (ppm)')
plt.ylabel('Tiempo Diario (min)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, '3_scatter_scroll_uso.png')) # <-- Guardado
plt.close() # <-- Cierre de figura

# Relación entre Likes Recibidos y Comentarios
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Likes_Received_Daily', y='Comments_Received_Daily', color='purple', alpha=0.5)
plt.title('Interacción: Likes vs Comentarios')
plt.savefig(os.path.join(carpeta_graficas, '4_scatter_likes_comentarios.png')) # <-- Guardado
plt.close() # <-- Cierre de figura

# INGENIERÍA DE CARACTERÍSTICAS Y TRANSFORMACIONES

print("\n--- INICIANDO TRANSFORMACIONES ---")

# Transformación Logarítmica (Corrección de Sesgo)
# La aplicamos al tiempo de uso para cerrar la brecha entre los usuarios con horas muy extensas
df['Log_Usage_Time'] = np.log1p(df['Daily_Usage_Time_min'])

# Verificación visual del cambio
plt.figure(figsize=(10, 5))
sns.histplot(df['Log_Usage_Time'], kde=True, color='green')
plt.title('Distribución del Tiempo de Uso (Log Transformada)')
plt.savefig(os.path.join(carpeta_graficas, '5_distribucion_uso_log.png')) # <-- Guardado
plt.close() # <-- Cierre de figura

# Matriz de Correlación entre las variables numericas
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación de Variables')
plt.savefig(os.path.join(carpeta_graficas, '6_matriz_correlacion.png')) # <-- Guardado
plt.close() # <-- Cierre de figura

# Decisión de eliminación basada en correlación, ya que si dos columnas tienen una correlacion
# elevada, podemos borrar una de ellas

# Codificación de Variables (Encoding)

# Label Encoding para el Target (Variable Objetivo)
le = LabelEncoder()
df['Target_Emotion'] = le.fit_transform(df['Emotional_State_Post_Usage'])
print("Target codificado: ", dict(zip(le.classes_, le.transform(le.classes_))))

# Binary Encoding (Manual) para Género
df['Is_Female'] = df['Gender'].apply(lambda x: 1 if x == 'female' else 0)

# One-Hot Encoding para Plataforma
df = pd.get_dummies(df, columns=['Platform'], prefix='Platform', drop_first=True)

# Escalado de Datos (StandardScaler)
# solo escalamos las variables numericas que se usan en la prediccion
scaler = StandardScaler()
cols_to_scale = ['Age', 'Daily_Usage_Time_min', 'Scroll_Rate_ppm', 
                 'Likes_Received_Daily', 'Posts_Per_Day', 'Log_Usage_Time']

# Verificamos que las columnas existan antes de escalar
cols_existentes = [c for c in cols_to_scale if c in df.columns]
df[cols_existentes] = scaler.fit_transform(df[cols_existentes])

print("\n--- DATOS ESCALADOS (Primeras 5 filas) ---")
print(df[cols_existentes].head())

# LIMPIEZA FINAL (PRE-ENTRENAMIENTO)

# Eliminar columnas originales de texto que ya fueron codificadas o que causan Data Leakage
cols_to_drop = ['Gender', 'Emotional_State_Post_Usage', 
                'Addiction_Level', 'Mental_Health_Index', 'FOMO_Score'] 
# Borramos Addiction Level, mental health index y FOMO Score ya que son valores
# que no podemos usar en tiempo real para la prediccion que resuelve nuestro problema, seran
# datos que no tendremos durante la ejecucion real
df_final = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

print("\n--- PIPELINE FINALIZADO ---")
print("Dimensiones del dataset final:", df_final.shape)
print(df_final.head())

# EXPORTAR EL DATASET PROCESADO
ruta_exportacion = os.path.join(carpeta_graficas, 'dataset_procesado_final.csv')
df_final.to_csv(ruta_exportacion, index=False)
print(f"\n✅ Dataset procesado guardado en: {ruta_exportacion}")