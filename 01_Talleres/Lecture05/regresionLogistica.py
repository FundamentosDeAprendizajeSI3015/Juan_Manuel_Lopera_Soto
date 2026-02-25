import pandas as pd
import numpy as np
import os
from scipy.stats import reciprocal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score

# Definamos el "random_state" para que los resultados sean reproducibles:
random_state = 42

# Cambiemos la fuente de las gráficas de matplotlib y creamos el directorio para guardar las graficas:
plt.rc('font', family='serif', size=12)
output_dir = "graficas_output_lecture5_logistica"
os.makedirs(output_dir, exist_ok=True)


# 1. CARGA Y LIMPIEZA DEL DATASET

# Cargamos los datos usando pandas
df = pd.read_csv('../Datos/HeartDiseaseData.csv')

# El dataset usa '?' para denotar valores nulos. Los reemplazamos por NaN
df = df.replace('?', np.nan)

# Eliminamos las filas con valores nulos (forma más rápida de limpiar para empezar)
df = df.dropna()

# Aseguramos que todas las columnas sean de tipo numérico
df = df.apply(pd.to_numeric)

# Separamos características (X) y objetivo (y)
# Binarizamos 'num': 0 es sano, cualquier valor > 0 es enfermo (1)
X = df.drop('num', axis=1).values
y = (df['num'] > 0).astype(int).values

# Separemos nuestros datos en conjuntos de entrenamiento y prueba:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)



# 2. PIPELINE Y ENTRENAMIENTO

# Definamos un pipeline con un solver adecuado para múltiples variables
lr_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=2000, solver='saga')) 
])

# Limitamos los grados polinomiales para evitar explosión combinatoria
param_distributions = {
    'poly__degree': [1, 2, 3],
    'classifier__C': reciprocal(1e-5, 1e5)
}

# Definamos nuestro modelo mediante RandomizedSearchCV:
lr = RandomizedSearchCV(
    lr_base,
    cv=4,
    param_distributions=param_distributions,
    n_iter=50,
    random_state=random_state,
    n_jobs=-1
)

# Entrenemos el modelo:
print("Entrenando el modelo de Regresión Logística...")
lr.fit(X_train, y_train)

# Obtengamos los mejores hiperparámetros encontrados para el modelo:
print(f'\nMejores parámetros: {lr.best_params_}')

# Predicciones de prueba
y_pred = lr.predict(X_test)

# Obtengamos la accuracy y el f1-score de prueba:
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1 score: {f1_score(y_test, y_pred):.4f}')


# 3. VISUALIZACIÓN

# Grafiquemos la matriz de confusión de los datos de prueba:
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sano (0)', 'Enfermo (1)'])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Matriz de Confusión - Detección Cardíaca')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()