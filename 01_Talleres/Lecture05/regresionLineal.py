import os
import numpy as np
import pandas as pd
from scipy.stats import reciprocal
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Crear directorio local para guardar las gráficas
output_dir = "graficas_output_lecture5_lineal"
os.makedirs(output_dir, exist_ok=True)

# Definimos el random state como la fuente que usaremos en matplotlib, esto tambien nos permite reproducir los mismos resultados con este valor
random_state = 42
np.random.seed(random_state)
plt.rc('font', family='serif', size=12)

# 2. Carga y preparación del dataset
df = pd.read_csv('../Datos/FIFA23Data.csv')

# Seleccionar las características numéricas a usar (X) y el objetivo (y)
features = [
    'Age', 'Overall', 'Potential', 'Pace Total', 'Shooting Total', 
    'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total'
]
target = 'Value(in Euro)'

# Filtrar datos nulos en estas columnas para evitar errores
df = df.dropna(subset=features + [target])

X = df[features].values
y = df[target].values / 1000000

# 3. Separación de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 4. Definición de pipelines
ridge_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

lasso_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(max_iter=40000, tol=0.01)) # Se tuvieron que aumentar la iteraciones, ya que durante los primeros entrenamientos,
    # Lasso no lograba converger al valor mas optimo
])

# Definimos un rango polinomial corto para evitar que la memoria explote
param_distributions = {
    'poly__degree': list(range(1, 3)),
    'regressor__alpha': reciprocal(1e-5, 1e3)
}

# 5. Modelos mediante RandomizedSearchCV
ridge = RandomizedSearchCV(
    ridge_base, cv=4, param_distributions=param_distributions,
    n_iter=20, random_state=random_state, n_jobs=-1
)

lasso = RandomizedSearchCV(
    lasso_base, cv=4, param_distributions=param_distributions,
    n_iter=20, random_state=random_state, n_jobs=-1
)

# 6. Entrenamiento
print("Entrenando modelo Ridge...")
ridge.fit(X_train, y_train)

print("Entrenando modelo LASSO...")
lasso.fit(X_train, y_train)

# 7. Evaluación y resultados
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

print('\n--- Resultados Ridge ---')
print(f"Mejores parámetros: {ridge.best_params_}")
print(f"R^2: {r2_score(y_test, y_pred_ridge)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_ridge):.2f} Millones de Euros")

print('\n--- Resultados LASSO ---')
print(f"Mejores parámetros: {lasso.best_params_}")
print(f"R^2: {r2_score(y_test, y_pred_lasso)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lasso):.2f} Millones de Euros")

# 8. Generación y guardado de gráficas (Valor Real vs Predicho)

# Gráfica Ridge
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_ridge, alpha=0.5, c='c', label='Predicciones Ridge')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2, label='Modelo Ideal (Real = Predicho)')
ax.set_xlabel('Valor Real del Jugador (Millones de Euros)')
ax.set_ylabel('Valor Predicho (Millones de Euros)')
ax.set_title('Ridge: Valor Real vs Predicho')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ridge_real_vs_predicho.png'))
plt.close() # Cierra la gráfica para que no se muestre en pantalla

# Gráfica LASSO
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_lasso, alpha=0.5, c='m', label='Predicciones LASSO')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2, label='Modelo Ideal (Real = Predicho)')
ax.set_xlabel('Valor Real del Jugador (Millones de Euros)')
ax.set_ylabel('Valor Predicho (Millones de Euros)')
ax.set_title('LASSO: Valor Real vs Predicho')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lasso_real_vs_predicho.png'))
plt.close()

print(f"\nSe han guardado las graficas en la carpeta: '{os.path.abspath(output_dir)}'")