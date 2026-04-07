# ==========================================================
# ULTIMATE REAL-WORLD PIPELINE (Adapted for Realistic Data)
# + Imputation & Categorical Encoding (OneHotEncoder)
# + DBSCAN & HDBSCAN Sweeps
# + Failure Analysis by 'Unidad'
# ==========================================================

import os
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, accuracy_score
from scipy.stats import mode

import hdbscan
import umap.umap_ as umap

# Suppress warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="whitegrid", context="talk")

OUTPUT_DIR = "clustering_udea_realista_graficos"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -----------------------------
# 1. DATA LOADING & PREP
# -----------------------------

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    y_true = None
    unidades = None
    
    # Store 'unidad' separately so we can analyze failures later
    if 'unidad' in df.columns:
        unidades = df['unidad'].copy()
        
    if 'label' in df.columns:
        print("[INFO] 'label' column found. Extracting for evaluation.")
        y_true = df['label'].values
        df = df.drop(columns=['label'])
        
    print(f"[INFO] Loaded realistic data: {df.shape[0]} samples, {df.shape[1]} raw features")
    return df, y_true, unidades

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    print("[INFO] Applying Advanced Preprocessing (Imputation, RobustScaler, OneHotEncoder)...")
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Pipeline for numbers: Fill missing with median -> Handle outliers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Pipeline for categories: Fill missing with most frequent -> Encode to binary columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine them
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(df)
    print(f"[INFO] Data expanded to {X_processed.shape[1]} dimensions after encoding.")
    return X_processed

# -----------------------------
# 2. K-DISTANCE IN HIGH-D
# -----------------------------

def k_distance_plot(X: np.ndarray, k: int = 5):
    print("[INFO] Calculating k-distance in High-D space...")
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Sorted observations')
    plt.ylabel(f'{k}-distance')
    plt.title('High-D k-distance plot (Find the Elbow for DBSCAN eps)')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "01_HighD_k_distance_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

# -----------------------------
# 3. 3D PROJECTION
# -----------------------------

def get_3d_projection(X: np.ndarray):
    print("[INFO] Projecting High-D data to 3D for visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
    return reducer.fit_transform(X)

# -----------------------------
# 4. EVALUATION MATH
# -----------------------------

def evaluate_clustering(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    
    y_pred_mapped = np.copy(y_pred)
    for cluster in np.unique(y_pred):
        if cluster != -1:
            mask = (y_pred == cluster)
            if mask.sum() > 0:
                majority_label = mode(y_true[mask], keepdims=True)[0][0]
                y_pred_mapped[mask] = majority_label
                
    acc = accuracy_score(y_true, y_pred_mapped)
    return ari, acc, y_pred_mapped

# -----------------------------
# 5. SWEEPS (WITH BEST MODEL TRACKING)
# -----------------------------

def run_dbscan_sweep(X_high_dim, X_3d, y_true, radiuses: list):
    print("\n[INFO] Running High-D DBSCAN Radius Sweep...")
    fig = plt.figure(figsize=(24, 14))
    
    best_ari = -1
    best_mapped = None
    best_name = ""

    for i, eps in enumerate(radiuses):
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X_high_dim)
        
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(labels == -1)
        
        metrics_text = ""
        if y_true is not None:
            ari, acc, y_pred_mapped = evaluate_clustering(y_true, labels)
            metrics_text = f"\nARI: {ari:.3f} | Acc: {acc:.1%}"
            
            # Track best performance for failure analysis
            if ari > best_ari:
                best_ari = ari
                best_mapped = y_pred_mapped
                best_name = f"DBSCAN (eps={eps})"
                
        palette = sns.color_palette("tab10", max(1, len(unique_labels)))
        colors = []
        for lbl in labels:
            if lbl == -1:
                colors.append((0.6, 0.6, 0.6)) 
            else:
                idx = np.where(unique_labels == lbl)[0][0]
                colors.append(palette[idx % len(palette)])

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, s=30, edgecolor='k', linewidth=0.2, alpha=0.9)
        ax.set_title(f"Radius (eps) = {eps}\nClusters: {n_clusters} | Noise: {noise_count}{metrics_text}", fontsize=14, pad=10)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    plt.suptitle("DBSCAN Radius Experiment in High Dimensions", fontsize=22, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_DBSCAN_Grid_Experiment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_ari, best_mapped, best_name

def run_hdbscan_sweep(X_high_dim, X_3d, y_true, parameter_pairs: list):
    print("\n[INFO] Running High-D HDBSCAN Parameter Sweep...")
    fig = plt.figure(figsize=(24, 14))
    
    best_ari = -1
    best_mapped = None
    best_name = ""

    for i, (m_cluster, m_samples) in enumerate(parameter_pairs):
        hdb = hdbscan.HDBSCAN(min_cluster_size=m_cluster, min_samples=m_samples)
        labels = hdb.fit_predict(X_high_dim)
        
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(labels == -1)
        
        metrics_text = ""
        if y_true is not None:
            ari, acc, y_pred_mapped = evaluate_clustering(y_true, labels)
            metrics_text = f"\nARI: {ari:.3f} | Acc: {acc:.1%}"
            
            if ari > best_ari:
                best_ari = ari
                best_mapped = y_pred_mapped
                best_name = f"HDBSCAN (mcs={m_cluster}, ms={m_samples})"
                
        palette = sns.color_palette("tab10", max(1, len(unique_labels)))
        colors = []
        for lbl in labels:
            if lbl == -1:
                colors.append((0.6, 0.6, 0.6))
            else:
                idx = np.where(unique_labels == lbl)[0][0]
                colors.append(palette[idx % len(palette)])

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, s=30, edgecolor='k', linewidth=0.2, alpha=0.9)
        ax.set_title(f"mcs={m_cluster} | ms={m_samples}\nClusters: {n_clusters} | Noise: {noise_count}{metrics_text}", fontsize=14, pad=10)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    plt.suptitle("HDBSCAN Parameter Sweep in High Dimensions", fontsize=22, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_HDBSCAN_Grid_Experiment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_ari, best_mapped, best_name

# -----------------------------
# 6. FAILURE ANALYSIS BY UNIDAD
# -----------------------------

def analyze_failures(y_true, y_pred_mapped, unidades, model_name):
    print(f"\n[INFO] Running Failure Analysis on the best model: {model_name}")
    
    # Create a DataFrame of the results
    df_err = pd.DataFrame({
        'unidad': unidades, 
        'true_label': y_true, 
        'predicted_label': y_pred_mapped
    })
    
    # Check where prediction failed (True if failed)
    df_err['failed'] = df_err['true_label'] != df_err['predicted_label']

    # Calculate Failure Rate (%) and Total Failures per Unidad
    failure_rates = df_err.groupby('unidad')['failed'].mean() * 100
    failure_counts = df_err.groupby('unidad')['failed'].sum()
    total_samples = df_err.groupby('unidad').size()

    stats = pd.DataFrame({
        'Total_Samples': total_samples,
        'Failure_Count': failure_counts,
        'Failure_Rate_%': failure_rates
    }).sort_values(by='Failure_Rate_%', ascending=False)
    
    print("\n[RESULT] Failure Statistics by Unidad:")
    print(stats.to_string())

    # Generate Bar Chart
    plt.figure(figsize=(14, 7))
    sns.barplot(x=stats.index, y=stats['Failure_Rate_%'], palette='Reds_r')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Prediction Failure Rate by Unidad\n(Model: {model_name})')
    plt.ylabel('Failure Rate (%)')
    plt.xlabel('Unidad')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "04_Failure_Analysis_Unidad.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Failure Analysis Plot to: {save_path}")

# -----------------------------
# 7. MAIN EXECUTION ROUTINE
# -----------------------------

def main():
    # Use the specific realistic dataset
    csv_path = '../../data/dataset_sintetico_FIRE_UdeA_realista.csv' 

    # 1. Load Data (Extracting 'unidad' for later)
    df, y_true, unidades = load_data(csv_path)
    
    # 2. Advanced Preprocessing (Scaling + Missing Values + Encoding)
    X_scaled_high_D = preprocess_data(df)

    # 3. K-Distance Math
    k_distance_plot(X_scaled_high_D, k=5)

    # 4. 3D Canvas
    X_umap_3d = get_3d_projection(X_scaled_high_D)

    if y_true is not None:
        # NOTE: Because OneHotEncoder adds many new dimension columns, distances expand.
        # Radiuses for DBSCAN are widened to account for the new High-D space.
        radiuses_to_test = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        db_ari, db_mapped, db_name = run_dbscan_sweep(X_scaled_high_D, X_umap_3d, y_true, radiuses_to_test)

        hdbscan_params = [
            (5, 2), (10, 2), (20, 2), 
            (10, 5), (20, 10), (30, 15)
        ]
        hdb_ari, hdb_mapped, hdb_name = run_hdbscan_sweep(X_scaled_high_D, X_umap_3d, y_true, hdbscan_params)
        
        # Determine the best performing model overall to run Failure Analysis
        if unidades is not None:
            if hdb_ari > db_ari:
                analyze_failures(y_true, hdb_mapped, unidades, hdb_name)
            else:
                analyze_failures(y_true, db_mapped, unidades, db_name)
        else:
            print("\n[WARNING] 'unidad' column not found, skipping Failure Analysis.")
            
        print("\n[ALL DONE] All experiments finished successfully! Check the output folder.")
    else:
        print("[ERROR] No 'label' column found.")

if __name__ == '__main__':
    main()