import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# 1. Configuración inicial
random_state = 42
plt.rc('font', family='serif', size=12)
output_dir = "graficas_clustering_realista"
os.makedirs(output_dir, exist_ok=True)

# 2. Cargar y preparar el dataset
df = pd.read_csv('../../data/dataset_sintetico_FIRE_UdeA_realista.csv')

# Separamos las caracteristicas y excluimos label para que sea no supervizado
X = df.drop(columns=['label'])

# Definimos las columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 3. Definir el pipeline de pre-procesamiento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Reducción a 2D para mantener las gráficas
# Transformamos los datos previamente solo para poder crear las coordenadas (X, Y) de las gráficas
X_preprocessed = preprocessor.fit_transform(X)
pca = PCA(n_components=2, random_state=random_state)
X_2d = pca.fit_transform(X_preprocessed)

# Grafica de los datos originales
fig, ax = plt.subplots()
ax.scatter(X_2d[:,0], X_2d[:,1])
fig.set_size_inches(5*1.6, 5)
plt.title("Visualización de Datos (PCA 2D)")
plt.savefig(os.path.join(output_dir, "01_datos_originales.png"), bbox_inches='tight')
plt.close()

# 5. K-Means con K = 2
clu_kmeans_2 = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("clustering", KMeans(n_clusters=2, random_state=random_state))
])

clu_kmeans_2.fit(X)

fig, ax = plt.subplots()
ax.scatter(X_2d[:,0], X_2d[:,1], c=clu_kmeans_2['clustering'].labels_, cmap='viridis')
fig.set_size_inches(5*1.6, 5)
plt.title("K-Means (K=2)")
plt.savefig(os.path.join(output_dir, "02_kmeans_k2.png"), bbox_inches='tight')
plt.close()

# 6. Curva de Inercia (Método del Codo)
inert = []
k_range = list(range(1, 11))
for k in k_range:
    clu_kmeans_k = Pipeline(steps=[
        ("preprocessor", preprocessor), 
        ("clustering", KMeans(n_clusters=k, random_state=random_state))
    ])
    clu_kmeans_k.fit(X)
    inert.append(clu_kmeans_k['clustering'].inertia_)

fig, ax = plt.subplots()
ax.plot(k_range, inert, marker='o')
fig.set_size_inches(5*1.6, 5)
plt.title("Curva de Inercia")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.savefig(os.path.join(output_dir, "03_curva_inercia.png"), bbox_inches='tight')
plt.close()

# 7. K-Means con K = 4
clu_kmeans_4 = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("clustering", KMeans(n_clusters=4, random_state=random_state))
])
clu_kmeans_4.fit(X)
print(f'Con K = 4: la inercia es {clu_kmeans_4["clustering"].inertia_}')

fig, ax = plt.subplots()
ax.scatter(X_2d[:,0], X_2d[:,1], c=clu_kmeans_4["clustering"].labels_, cmap='viridis')
fig.set_size_inches(5*1.6, 5)
plt.title("K-Means (K=4)")
plt.savefig(os.path.join(output_dir, "04_kmeans_k4.png"), bbox_inches='tight')
plt.close()

# 8. DBSCAN
clu_dbscan = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("clustering", DBSCAN(eps=2.5, min_samples=3))
])
clu_dbscan.fit(X)

fig, ax = plt.subplots()
ax.scatter(X_2d[:,0], X_2d[:,1], c=clu_dbscan["clustering"].labels_, cmap='plasma')
fig.set_size_inches(5*1.6, 5)
plt.title("DBSCAN (eps=2.5, min_samples=3)")
plt.savefig(os.path.join(output_dir, "05_dbscan.png"), bbox_inches='tight')
plt.close()

clusters_unicos, conteos = np.unique(clu_dbscan["clustering"].labels_, return_counts=True)
for cluster_id, count in zip(clusters_unicos, conteos):
    print(f"Cluster {cluster_id}: {count} elementos")