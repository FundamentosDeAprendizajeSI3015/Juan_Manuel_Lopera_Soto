import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import plot_tree

data = pd.read_csv('../Datos/FIFA23Data.csv')

columnas_a_eliminar = [
    'Known As', 'Full Name', 'Image Link', 'National Team Image Link',
    'National Team Name', 'Club Name', 'Positions Played', 'Club Position', 
    'National Team Position', 'ST Rating', 'LW Rating', 'LF Rating', 'CF Rating', 
    'RF Rating', 'RW Rating', 'CAM Rating', 'LM Rating', 'CM Rating', 'RM Rating', 
    'LWB Rating', 'CDM Rating', 'RWB Rating', 'LB Rating', 'CB Rating', 'RB Rating', 'GK Rating'
]
data.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

X = data.drop(columns='Best Position')
y = data['Best Position']

cat_cols = X.select_dtypes(include=['object', 'string']).columns
num_cols = X.select_dtypes(include=np.number).columns

categorical_transformer = Pipeline(
    steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, cat_cols)],
    remainder='passthrough'
)

rf_base = RandomForestClassifier(random_state=42)
hgb_base = HistGradientBoostingClassifier(random_state=42)

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_base)])
pipeline_hgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', hgb_base)])

param_grid_rf = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [6, 9, 12],
    'classifier__min_samples_leaf': [10, 50]
}

param_grid_hgb = {
    'classifier__max_iter': [50, 100],
    'classifier__max_depth': [6, 9, 12],
    'classifier__min_samples_leaf': [10, 50]
}

rf = GridSearchCV(pipeline_rf, cv=3, param_grid=param_grid_rf, n_jobs=-1)
hgb = GridSearchCV(pipeline_hgb, cv=3, param_grid=param_grid_hgb, n_jobs=-1)

print("Entrenando Random Forest...")
rf.fit(X_train, y_train)

print("Entrenando HistGradientBoosting...")
hgb.fit(X_train, y_train)

mejor_rf = rf.best_estimator_

# --- 3. VISUALIZACIONES ---

# Configuración de estilo y carpeta para graficos
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 10})
output_dir = "graficas_output_lecture6"
os.makedirs(output_dir, exist_ok=True)

# A. Matriz de Confusión
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay.from_estimator(
    mejor_rf, X_test, y_test, 
    ax=ax, cmap='viridis', xticks_rotation='vertical'
)
plt.title('Matriz de Confusión - Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_RF.png'))
plt.close()

# B. Importancia de las Características (Feature Importance)
# Extraemos los nombres de las columnas procesadas
preprocesador_entrenado = mejor_rf.named_steps['preprocessor']

nombres_cat = preprocesador_entrenado.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
nombres_columnas = list(nombres_cat) + list(num_cols)

importancias = mejor_rf.named_steps['classifier'].feature_importances_
indices = np.argsort(importancias)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importancias[indices], color='cyan', align='center')
plt.yticks(range(len(indices)), [nombres_columnas[i] for i in indices])
plt.xlabel('Importancia Relativa')
plt.title('Top 15 Atributos más importantes para predecir la posición')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# C. Graficar un Árbol de Decisión Individual
arbol_individual = mejor_rf.named_steps['classifier'].estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(
    arbol_individual, 
    feature_names=nombres_columnas, 
    class_names=mejor_rf.classes_, 
    filled=True, 
    max_depth=3, 
    fontsize=8,
    proportion=True
)
plt.title('Estructura de un Árbol de Decisión (Profundidad truncada a 3 para visualización)')
plt.savefig(os.path.join(output_dir, 'singular_decision_tree.png'))
plt.close()