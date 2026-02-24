import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


# 1. LECTURA DE DATOS

print("--- 1. Lectura de Datos ---")
# Cargamos el dataset
df = pd.read_csv('../Datos/Titanic-Dataset.csv')

# Mostramos una vista general del dataset
print(df.head())
print("\nInformación del dataset:")
df.info()


# 2. CÁLCULO DE MEDIDAS ESTADÍSTICAS

print("\n--- 2. Cálculo de Medidas Estadísticas ---")
# Calculamos las medidas de tendencia central, dispersión y posición
stats = df.describe()
print("Estadísticas descriptivas del dataset:")
print(stats)


# 3. GRÁFICOS DE VISUALIZACIÓN Y DETECCIÓN DE OUTLIERS

print("\n--- 3. Generando Gráficos de Visualización ---")
# Configuramos el estilo de los graficos y la carpeta donde seran guardados
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

output_dir = "graficas_output_lecture4"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta creada: {output_dir}")
else:
    print(f"Las gráficas se guardarán en la carpeta existente: {output_dir}")

# Gráfico 1: Histogramas para ver las distribuciones
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Age'].dropna(), kde=True, color="skyblue")
plt.title("Distribución de Edad")
plt.subplot(1, 2, 2)
sns.histplot(df['Fare'], kde=True, color="salmon")
plt.title("Distribución de Tarifa (Fare)")
save_path = os.path.join(output_dir, 'histogramas_distribucion.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# Gráfico 2: Gráficos de dispersión para ver los outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df.index, y=df['Age'], color="skyblue", alpha=0.6)
plt.title("Dispersión de Edad")
plt.subplot(1, 2, 2)
sns.scatterplot(x=df.index, y=df['Fare'], color="salmon", alpha=0.6)
plt.title("Dispersión de Tarifa")
save_path = os.path.join(output_dir, 'dispersion_outliers.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# Gráfico 3: Boxplot para confirmar y ver los outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color="skyblue")
plt.title("Boxplot de Edad")
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'], color="salmon")
plt.title("Boxplot de Tarifa")
save_path = os.path.join(output_dir, 'boxplot_outliers.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")


# 4. PREPROCESAMIENTO DE LOS DATOS

print("\n--- 4. Preprocesamiento de los Datos ---")

# a) Llenamos la edad con la mediana y el puerto de embarque con la moda
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# b) Borramos las columnas que solo sirven para identificar y las que tienen muchos nulos
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
print("Columnas irrelevantes (PassengerId, Name, Ticket, Cabin) eliminadas.")

# c) Transformamos las variables categóricas a numéricas
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'].astype(str))
print("Variables categóricas codificadas a formato numérico.")

# d) Análisis de Correlación y Mapa de Calor en Grafica
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor de Correlaciones de las Características")
save_path = os.path.join(output_dir, 'heatmap_correlaciones.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# e) Estandarizamos los valores numericos por medio de un StandardScaler
scaler = StandardScaler()
cols_to_scale = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
print("Variables numéricas escaladas correctamente.")


# 5. GENERACIÓN DEL DATASET LISTO

print("\n--- 5. Generación del Dataset Listo ---")
output_csv = 'Titanic-Dataset-Ready.csv'
df.to_csv(output_csv, index=False)

print("\nPreprocesamiento finalizado con éxito.")
print(f"  -> Dataset listo para modelado guardado como: {output_csv}")
print("\nVista final del dataset procesado:")
print(df.head())
