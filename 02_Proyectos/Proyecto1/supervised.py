import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Configuración inicial
sns.set(style="whitegrid")
carpeta_graficas = "graficas_exportadas"
os.makedirs(carpeta_graficas, exist_ok=True)

df = pd.read_csv(os.path.join(carpeta_graficas, 'dataset_procesado_final.csv'))
y = df['Target_Emotion']
X = df.drop(columns=['Target_Emotion'])

print("--- INICIANDO APRENDIZAJE SUPERVISADO ---")

# 1. División de los datos (Train / Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=None)

print(f"Datos de Entrenamiento: {X_train.shape[0]} usuarios")
print(f"Datos de Prueba (Test): {X_test.shape[0]} usuarios")

# ==========================================
# MODELO 1: REGRESIÓN LOGÍSTICA (Baseline)
# ==========================================
# max_iter alto para asegurar que las matemáticas converjan
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)

# Matriz de Confusión Logística
plt.figure(figsize=(8, 6))
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión: Regresión Logística')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Emoción Real')
plt.savefig(os.path.join(carpeta_graficas, '12_matriz_logistica.png'))
plt.close()

# ==========================================
# MODELO 2: ÁRBOL DE DECISIÓN
# ==========================================
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='entropy')
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

# Matriz de Confusión Árbol
plt.figure(figsize=(8, 6))
cm_tree = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusión: Árbol de Decisión')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Emoción Real')
plt.savefig(os.path.join(carpeta_graficas, '13_matriz_arbol.png'))
plt.close()

# ------------------------------------------
# INTERPRETABILIDAD: DIBUJAR EL ÁRBOL
# ------------------------------------------
plt.figure(figsize=(15, 10))
plot_tree(tree_model, filled=True, feature_names=X.columns, 
          class_names=[str(c) for c in np.unique(y)], rounded=True, fontsize=10)
plt.title('Estructura de Reglas del Árbol de Decisión')
plt.savefig(os.path.join(carpeta_graficas, '14_estructura_arbol.png'))
plt.close()

# ------------------------------------------
# FEATURE IMPORTANCE (Importancia de Características)
# ------------------------------------------
importancias = tree_model.feature_importances_
df_importancias = pd.DataFrame({'Característica': X.columns, 'Importancia': importancias})
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False).head(5)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_importancias, x='Importancia', y='Característica', palette='Reds_r')
plt.title('Top 5 Variables para predecir Estado Emocional')
plt.savefig(os.path.join(carpeta_graficas, '15_importancia_variables.png'))
plt.close()

# ==========================================
# REPORTES DE RENDIMIENTO
# ==========================================
print("\n--- REPORTE REGRESIÓN LOGÍSTICA ---")
print(classification_report(y_test, y_pred_log, zero_division=0))

print("\n--- REPORTE ÁRBOL DE DECISIÓN ---")
print(classification_report(y_test, y_pred_tree, zero_division=0))

print("\nGráficas de evaluación y matrices guardadas exitosamente.")