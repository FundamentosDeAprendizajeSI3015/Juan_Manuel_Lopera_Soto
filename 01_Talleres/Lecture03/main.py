import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


# 1. LECTURA DE DATOS

print("--- 1. Lectura de Datos ---")
# Cargamos el dataset
df = pd.read_csv('../Datos/iris_data.csv')

# Mostramos una vista general del dataset
print(df.head())
print("\nInformación del dataset:")
df.info()

# Borramos la columna ID ya que solo sirve para identificar, no para predecir
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)


# 2. INGENIERÍA DE CARACTERÍSTICAS

print("\n--- 2. Ingeniería de Características ---")
# Creamos dos variables nuevas que nos ayuden a aproximar el area del sepalo y el petalo
df['SepalAreaCm2'] = df['SepalLengthCm'] * df['SepalWidthCm']
df['PetalAreaCm2'] = df['PetalLengthCm'] * df['PetalWidthCm']

print("Dataset tras añadir nuevas características (Áreas):")
print(df.head())


# 3. GRÁFICOS DE VISUALIZACIÓN

print("\n--- 3. Generando Gráficos de Visualización ---")
# Configuramos el estilo de los graficos y la carpeta donde seran guardados
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

output_dir = "graficas_output_lecture3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta creada: {output_dir}")
else:
    print(f"Las gráficas se guardarán en la carpeta existente: {output_dir}")

# Gráfico 1: Pairplot para ver las relaciones entre las variables (incluyendo las nuevas)
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue="Species", markers=["o", "s", "D"], palette="Set2")
plt.suptitle("Relaciones entre características por Especie", y=1.02)
save_path = os.path.join(output_dir, 'pairplot_relaciones_variables.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# Gráfico 2: Mapa de calor (Heatmap) de correlaciones
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor de Correlaciones de las Características")
save_path = os.path.join(output_dir, 'heatpmap_correlaciones.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")

# Gráfico 3: Boxplot para detectar posibles outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_cols, orient="h", palette="pastel")
plt.title("Distribución de variables y detección de Outliers")
save_path = os.path.join(output_dir, 'boxplot_outliers.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Gráfica guardada: {save_path}")


# 4. PREPROCESAMIENTO DE LOS DATOS

print("\n--- 4. Preprocesamiento de los Datos ---")

# a) Separar las características (X) y la variable objetivo (y)
X = df.drop('Species', axis=1)
y = df['Species']

# b) Codificar la variable objetivo (Label Encoding)
# Transformamos las clases de texto a numericas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Clases codificadas: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# c) División de datos en conjunto de Entrenamiento y Prueba
# Dejamos un 20% para test y 80% para entrenamiento. 
# Usamos stratify=y_encoded para mantener la proporción de las clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# d) Escalamiento de características (Feature Scaling)
# Estandarizamos los datos para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()

# Ajustamos el scaler solo con los datos de entrenamiento para evitar que se presente data leakage
X_train_scaled = scaler.fit_transform(X_train)
# Transformamos los datos de prueba
X_test_scaled = scaler.transform(X_test)

print("\nPreprocesamiento finalizado con éxito.")
print(f"  -> Tamaño de datos de entrenamiento (X_train_scaled): {X_train_scaled.shape}")
print(f"  -> Tamaño de datos de prueba (X_test_scaled): {X_test_scaled.shape}")
