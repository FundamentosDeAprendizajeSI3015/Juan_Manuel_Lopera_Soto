import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Estilos
sns.set(style="whitegrid")
carpeta_graficas = "graficas_exportadas"
os.makedirs(carpeta_graficas, exist_ok=True)

# 1. Cargar Datos
df = pd.read_csv(os.path.join(carpeta_graficas, 'dataset_procesado_final.csv'))
y_original = df['Target_Emotion']
X = df.drop(columns=['Target_Emotion'])

# 2. Reducción de Dimensionalidad (PCA)
# Pasamos de 19 dimensiones a 2 dimensiones para poder hacer gráficas X, Y
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Añadimos las coordenadas 2D a un DataFrame para graficar fácilmente
df_pca = pd.DataFrame(X_pca, columns=['Componente_1', 'Componente_2'])

print("--- INICIANDO APRENDIZAJE NO SUPERVISADO ---")

# ==========================================
# MODELO 1: K-MEANS
# ==========================================
# Asumimos 3 grupos comportamentales (Bajo, Medio, Alto)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_pca['KMeans_Cluster'] = kmeans.fit_predict(X)

# Calculamos el Silhouette Score (Mide qué tan buenos son los grupos de -1 a 1)
sil_kmeans = silhouette_score(X, df_pca['KMeans_Cluster'])
print(f"Calidad de K-Means (Silhouette Score): {sil_kmeans:.2f}")

# Gráfica K-Means
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='Componente_1', y='Componente_2', 
                hue='KMeans_Cluster', palette='viridis', s=100)
# Dibujar los centroides
centroides_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.title('Clustering con K-Means (Proyección PCA 2D)')
plt.legend()
plt.savefig(os.path.join(carpeta_graficas, '7_kmeans_clusters.png'))
plt.close()

# ==========================================
# MODELO 2: DBSCAN
# ==========================================
dbscan = DBSCAN(eps=0.5, min_samples=6)
df_pca['DBSCAN_Cluster'] = dbscan.fit_predict(X_pca)

# Gráfica DBSCAN
plt.figure(figsize=(8, 6))
# Nota: En DBSCAN, el cluster "-1" significa "Ruido / Anomalía" (Usuarios extremos)
sns.scatterplot(data=df_pca, x='Componente_1', y='Componente_2', 
                hue='DBSCAN_Cluster', palette='Set1', s=100)
plt.title('Clustering con DBSCAN (Ruido = -1)')
plt.legend()
plt.savefig(os.path.join(carpeta_graficas, '8_dbscan_clusters.png'))
plt.close()

# Guardamos los resultados para el Paso 5
df_pca['Target_Original'] = y_original
df_pca.to_csv(os.path.join(carpeta_graficas, 'dataset_con_clusters.csv'), index=False)

print("\nGráficas de K-Means y DBSCAN guardadas exitosamente en la carpeta.")
print("Dataset con etiquetas de cluster guardado para validación posterior.")