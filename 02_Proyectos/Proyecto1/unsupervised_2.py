import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Configuración inicial
sns.set(style="whitegrid")
carpeta_graficas = "graficas_exportadas"
df = pd.read_csv(os.path.join(carpeta_graficas, 'dataset_procesado_final.csv'))
y_original = df['Target_Emotion']
X = df.drop(columns=['Target_Emotion'])

# --- LA SOLUCIÓN AL ERROR ---
# Convertimos todo a números decimales para que scipy y skfuzzy no se quejen
X = X.astype(float)
# ----------------------------

# PCA para graficar en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=['Componente_1', 'Componente_2'])

print("--- INICIANDO FUZZY C-MEANS Y CLUSTERING JERÁRQUICO ---")

# ==========================================
# MODELO 3: FUZZY C-MEANS
# ==========================================
X_fuzz = X.T.values # Ahora sí, transpuesto como decimales puros

# c=3 (queremos 3 grupos), m=2 (coeficiente de "difuminado" estándar)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fuzz, c=3, m=2, error=0.005, maxiter=1000, init=None
)

df_pca['Fuzzy_Cluster'] = np.argmax(u, axis=0)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='Componente_1', y='Componente_2', 
                hue='Fuzzy_Cluster', palette='magma', s=100)
plt.title('Fuzzy C-Means (Clúster por Probabilidad Máxima)')
plt.savefig(os.path.join(carpeta_graficas, '9_fuzzy_cmeans.png'))
plt.close()

# ==========================================
# MODELO 4: CLUSTERING JERÁRQUICO Y DENDROGRAMA
# ==========================================
plt.figure(figsize=(10, 6))
Z = linkage(X, method='ward') 
dendrogram(Z)
plt.title('Dendrograma: Relación Jerárquica entre Usuarios')
plt.xlabel('Índice del Usuario')
plt.ylabel('Distancia Euclidiana (Comportamental)')
plt.savefig(os.path.join(carpeta_graficas, '10_dendrograma.png'))
plt.close()

jerarquico = AgglomerativeClustering(n_clusters=3, linkage='ward')
df_pca['Hierarchical_Cluster'] = jerarquico.fit_predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='Componente_1', y='Componente_2', 
                hue='Hierarchical_Cluster', palette='coolwarm', s=100)
plt.title('Clustering Jerárquico (Aglomerativo)')
plt.savefig(os.path.join(carpeta_graficas, '11_jerarquico.png'))
plt.close()

# Guardar todo
df_consolidado = pd.read_csv(os.path.join(carpeta_graficas, 'dataset_con_clusters.csv'))
df_consolidado['Fuzzy_Cluster'] = df_pca['Fuzzy_Cluster']
df_consolidado['Hierarchical_Cluster'] = df_pca['Hierarchical_Cluster']
df_consolidado.to_csv(os.path.join(carpeta_graficas, 'dataset_clusters_completo.csv'), index=False)

print("\n¡Gráficas generadas exitosamente!")