# ==========================================================
# ULTIMATE HIGH-DIMENSIONAL MASTER PIPELINE
# Contains: Preprocessing, K-Distance, UMAP 3D Projection, 
# DBSCAN Radius Sweep, and HDBSCAN Parameter Sweep.
# ==========================================================

import os
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, accuracy_score
from scipy.stats import mode

import hdbscan
import umap.umap_ as umap

# Suppress UMAP warnings for cleaner terminal output
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
sns.set(style="whitegrid", context="talk")

OUTPUT_DIR = "clustering_udea_no_realista_graficos"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -----------------------------
# 1. DATA LOADING & PREP
# -----------------------------

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    y_true = None
    if 'label' in df.columns:
        print("[INFO] 'label' column found. Extracting for evaluation.")
        y_true = df['label'].values
        df = df.drop(columns=['label'])
    print(f"[INFO] Loaded data: {df.shape[0]} samples, {df.shape[1]} features (7D)")
    return df, y_true

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    print("[INFO] Applying RobustScaler to handle financial outliers...")
    scaler = RobustScaler()
    return scaler.fit_transform(df.values)

# -----------------------------
# 2. K-DISTANCE IN HIGH-D
# -----------------------------

def k_distance_plot(X: np.ndarray, k: int = 5):
    print("[INFO] Calculating k-distance in 7D space...")
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Sorted observations')
    plt.ylabel(f'{k}-distance (7D Euclidean)')
    plt.title('7D k-distance plot (Find the Elbow for DBSCAN eps)')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "01_7D_k_distance_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved K-Distance Plot: {save_path}")

# -----------------------------
# 3. 3D PROJECTION (VISUAL CANVAS)
# -----------------------------

def get_3d_projection(X: np.ndarray):
    print("[INFO] Projecting 7D data down to 3D for visualization...")
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
    return ari, acc

# -----------------------------
# 5. DBSCAN RADIUS SWEEP
# -----------------------------

def run_dbscan_sweep(X_high_dim, X_3d, y_true, radiuses: list, filename: str):
    print("\n[INFO] Running 7D DBSCAN Radius Sweep...")
    fig = plt.figure(figsize=(24, 14))
    
    for i, eps in enumerate(radiuses):
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X_high_dim)
        
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(labels == -1)
        
        metrics_text = ""
        if y_true is not None:
            ari, acc = evaluate_clustering(y_true, labels)
            metrics_text = f"\nARI: {ari:.3f} | Acc: {acc:.1%}"
            
        palette = sns.color_palette("tab10", max(1, len(unique_labels)))
        colors = []
        for lbl in labels:
            if lbl == -1:
                colors.append((0.6, 0.6, 0.6)) # Strict 3-value RGB for grey noise
            else:
                idx = np.where(unique_labels == lbl)[0][0]
                colors.append(palette[idx % len(palette)])

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, s=30, edgecolor='k', linewidth=0.2, alpha=0.9)
        
        title = f"7D Radius (eps) = {eps}\nClusters: {n_clusters} | Noise: {noise_count}{metrics_text}"
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.suptitle("DBSCAN Radius Experiment in 7D (Visualized in 3D)", fontsize=22, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved DBSCAN Sweep Plot to: {save_path}")

# -----------------------------
# 6. HDBSCAN PARAMETER SWEEP
# -----------------------------

def run_hdbscan_sweep(X_high_dim, X_3d, y_true, parameter_pairs: list, filename: str):
    print("\n[INFO] Running 7D HDBSCAN Parameter Sweep...")
    fig = plt.figure(figsize=(24, 14))
    
    for i, (min_cluster_size, min_samples) in enumerate(parameter_pairs):
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = hdb.fit_predict(X_high_dim)
        
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        noise_count = np.sum(labels == -1)
        
        metrics_text = ""
        if y_true is not None:
            ari, acc = evaluate_clustering(y_true, labels)
            metrics_text = f"\nARI: {ari:.3f} | Acc: {acc:.1%}"
            
        palette = sns.color_palette("tab10", max(1, len(unique_labels)))
        colors = []
        for lbl in labels:
            if lbl == -1:
                colors.append((0.6, 0.6, 0.6)) # Strict 3-value RGB for grey noise
            else:
                idx = np.where(unique_labels == lbl)[0][0]
                colors.append(palette[idx % len(palette)])

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, s=30, edgecolor='k', linewidth=0.2, alpha=0.9)
        
        title = f"min_cluster_size={min_cluster_size} | min_samples={min_samples}\nClusters: {n_clusters} | Noise: {noise_count}{metrics_text}"
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.suptitle("HDBSCAN Parameter Sweep in 7D (Visualized in 3D)", fontsize=22, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved HDBSCAN Sweep Plot to: {save_path}")

# -----------------------------
# 7. MAIN EXECUTION ROUTINE
# -----------------------------

def main():
    csv_path = '../../data/dataset_sintetico_FIRE_UdeA.csv' 

    # 1. Load Data & Scale
    df, y_true = load_data(csv_path)
    X_scaled_7D = preprocess_data(df)

    # 2. Check the Math (K-Distance)
    k_distance_plot(X_scaled_7D, k=5)

    # 3. Create the 3D Canvas
    X_umap_3d = get_3d_projection(X_scaled_7D)

    if y_true is not None:
        # --- EXPERIMENT 1: DBSCAN RADIUS SWEEP ---
        radiuses_to_test = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
        run_dbscan_sweep(X_scaled_7D, X_umap_3d, y_true, radiuses_to_test, "02_7D_DBSCAN_Grid_Experiment")

        # --- EXPERIMENT 2: HDBSCAN PARAMETER SWEEP ---
        # Moving from aggressive (small numbers) to conservative (large numbers)
        hdbscan_params = [
            (5, 2),   # Aggressive
            (10, 2), 
            (20, 2), 
            (10, 5),  # Moderate
            (20, 10), # Strict
            (30, 15)  # Very Conservative
        ]
        run_hdbscan_sweep(X_scaled_7D, X_umap_3d, y_true, hdbscan_params, "03_7D_HDBSCAN_Grid_Experiment")
        
        print("\n[ALL DONE] All experiments finished successfully! Check the 'plots_ultimate_pipeline' folder.")
    else:
        print("[ERROR] No 'label' column found.")

if __name__ == '__main__':
    main()