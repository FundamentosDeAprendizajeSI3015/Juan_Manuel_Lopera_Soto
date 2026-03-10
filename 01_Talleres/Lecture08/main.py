import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización de tablas en consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Silenciar advertencias
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Machine Learning & Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             brier_score_loss, log_loss, precision_score, 
                             recall_score, f1_score, confusion_matrix,
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.tree import plot_tree
import shap

# Intento de cargar UMAP
try:
    import umap.umap_ as umap
    UMAP_DISPONIBLE = True
except ImportError:
    UMAP_DISPONIBLE = False

# ====================================================================
# 1. CARGA DE DATOS Y PREPARACIÓN
# ====================================================================
print("Iniciando Pipeline Integral (Reporte Legible + 8 Gráficas)...")
df = pd.read_csv('../Datos/dataset_sintetico_FIRE_UdeA_realista.csv')

target = 'label'
cat_features = ['unidad']
cols_excluir = [target, 'anio']
num_features = [col for col in df.columns if col not in cols_excluir + cat_features]

X = df.drop(columns=cols_excluir)
y = df[target]

# Splits 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# ====================================================================
# 2. PIPELINE DE MODELADO
# ====================================================================
num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])

# Modelo con regularización para evitar overfitting excesivo
gb_model = GradientBoostingClassifier(
    n_estimators=50, 
    learning_rate=0.02, 
    max_depth=2, 
    min_samples_leaf=5, 
    random_state=42
)
pipeline_final = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', gb_model)])
pipeline_final.fit(X_train, y_train)

# ====================================================================
# 3. CÁLCULO DE MÉTRICAS
# ====================================================================
def get_metrics_full(model, X_set, y_set, name):
    probs = model.predict_proba(X_set)[:, 1]
    preds = model.predict(X_set)
    tn, fp, fn, tp = confusion_matrix(y_set, preds, labels=[0, 1]).ravel()
    
    return {
        'split': name,
        'n': len(y_set),
        'prevalencia': np.mean(y_set),
        'roc_auc': roc_auc_score(y_set, probs),
        'pr_auc': average_precision_score(y_set, probs),
        'brier': brier_score_loss(y_set, probs),
        'log_loss': log_loss(y_set, probs),
        'precision': precision_score(y_set, preds, zero_division=0),
        'recall': recall_score(y_set, preds, zero_division=0),
        'f1': f1_score(y_set, preds, zero_division=0),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

results = [
    get_metrics_full(pipeline_final, X_train, y_train, 'train'),
    get_metrics_full(pipeline_final, X_valid, y_valid, 'valid'),
    get_metrics_full(pipeline_final, X_test,  y_test,  'test')
]

df_res = pd.DataFrame(results)

print("\n" + "="*50)
print("TABLA DE RESULTADOS FINALES")
print("="*50)
# Formato legible con columnas alineadas
print(df_res.to_string(index=False))
print("="*50 + "\n")

# ====================================================================
# 4. GENERACIÓN DE LAS 8 GRÁFICAS MAESTRAS
# ====================================================================
print("Generando archivos PNG...")
sns.set_theme(style="whitegrid")

prep = pipeline_final.named_steps['preprocessor']
model = pipeline_final.named_steps['classifier']
X_train_trans = prep.transform(X_train)
X_test_trans = prep.transform(X_test)
cat_out = prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
f_names = num_features + list(cat_out)

# 01. UMAP
if UMAP_DISPONIBLE:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embed = reducer.fit_transform(X_train_trans)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embed[:, 0], y=embed[:, 1], hue=y_train, palette='coolwarm', s=80)
    plt.title("01. Proyección UMAP")
    plt.savefig('01_umap.png', dpi=300); plt.close()

# 02. Matriz de Confusión
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, pipeline_final.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sano", "Tensión"])
disp.plot(cmap='Blues', ax=plt.gca()); plt.title("02. Matriz de Confusión (Test)")
plt.savefig('02_conf_matrix.png', dpi=300); plt.close()

# 03. ROC Curva
plt.figure(figsize=(7, 5))
fpr_te, tpr_te, _ = roc_curve(y_test, pipeline_final.predict_proba(X_test)[:, 1])
plt.plot(fpr_te, tpr_te, label=f"Test AUC: {roc_auc_score(y_test, pipeline_final.predict_proba(X_test)[:, 1]):.4f}", color='darkorange')
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("03. Curva ROC (Test)")
plt.savefig('03_roc_curve.png', dpi=300); plt.close()

# 04. Precision-Recall
plt.figure(figsize=(7, 5))
p_te, r_te, _ = precision_recall_curve(y_test, pipeline_final.predict_proba(X_test)[:, 1])
plt.plot(r_te, p_te, color='purple', label=f"PR AUC: {average_precision_score(y_test, pipeline_final.predict_proba(X_test)[:, 1]):.4f}")
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.title("04. Curva Precision-Recall")
plt.savefig('04_pr_curve.png', dpi=300); plt.close()

# 05. Calibración
plt.figure(figsize=(7, 5))
prob_t, prob_p = calibration_curve(y_train, pipeline_final.predict_proba(X_train)[:, 1], n_bins=5)
plt.plot(prob_p, prob_t, marker='o', label='Train Calibrado')
plt.plot([0,1],[0,1],'--', color='gray'); plt.legend(); plt.title("05. Curva de Calibración")
plt.savefig('05_calibration.png', dpi=300); plt.close()

# 06. Feature Importance
plt.figure(figsize=(10, 6))
importances = pd.DataFrame({'f': f_names, 'i': model.feature_importances_}).sort_values('i', ascending=False)
sns.barplot(data=importances.head(10), x='i', y='f', palette='mako', hue='f', legend=False)
plt.title("06. Top 10 Variables (Importancia)"); plt.savefig('06_importance.png', dpi=300); plt.close()

# 07. SHAP
plt.figure(figsize=(10, 6))
explainer = shap.TreeExplainer(model)
shap_v = explainer.shap_values(X_test_trans)
shap.summary_plot(shap_v, X_test_trans, feature_names=f_names, show=False)
plt.title("07. Análisis SHAP"); plt.savefig('07_shap.png', dpi=300); plt.close()

# 08. Árbol de Decisión
plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0, 0], feature_names=f_names, max_depth=2, filled=True, rounded=True, fontsize=12)
plt.title("08. Reglas del Primer Árbol"); plt.savefig('08_tree.png', dpi=300); plt.close()

print("¡Proceso finalizado con éxito!")