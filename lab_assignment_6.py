from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


housing = fetch_california_housing(as_frame=True)
df = housing.frame

#using this to check all features names
print(df.head())
print(housing.feature_names)

X = df[['MedInc','Latitude','Longitude']]

#using a scaler
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)

# finding best k, best score and visualizing silhouette score against K
k_range = range(2,11)
sil_scores = []

for K in k_range:
    k_means = KMeans(n_clusters=K,n_init=10,random_state=42)
    k_means.fit(X_scaled)
    labels = k_means.predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    
print(f'silhouette score array : {sil_scores}')

best_idx = np.argmax(sil_scores)       #This gives the which position has max values    
best_k = list(k_range)[best_idx]         # This gives which K values has max value
best_score = sil_scores[best_idx]

print(f'best K is :{best_k}')
print(f"best score is: {best_score}")

k_values = list(k_range)                 # make it a list for plotting

# visualization for silhouette score against K
plt.figure(figsize=(18, 10))
plt.scatter(k_values, sil_scores, s=120, color='blue')
plt.plot(k_values, sil_scores, linewidth=3, color='green')
plt.scatter([best_k], [best_score], s=300, color='red', edgecolor='black', zorder=5)

plt.text(best_k + 0.1, best_score - 0.02,
         f"best K = {best_k}\nscore = {best_score:.3f}",
         fontsize=22, va="top", ha="left",
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

plt.xticks(k_values, fontsize=20, rotation=35)
plt.yticks(fontsize=20)
plt.xlabel("K (number of clusters)", fontsize=28, labelpad=10)
plt.ylabel("Silhouette score", fontsize=28, labelpad=10)
plt.title("Silhouette Score vs K (K-Means Clustering)", fontsize=38, fontweight='bold', pad=30)
plt.grid(True, alpha=0.4)
plt.tight_layout(pad=4)
plt.show()

# Visualizing clusters with different K values

k_values_to_plot = [2, 3, 4] 
plt.figure(figsize=(24, 8))  

for i, k in enumerate(k_values_to_plot, 1):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    plt.subplot(1, len(k_values_to_plot), i)
    plt.scatter(X['Longitude'], X['Latitude'], c=labels, s=15, cmap='viridis')

    plt.title(f"K = {k}", fontsize=40, fontweight='bold', pad=20)
    plt.xlabel("Longitude", fontsize=28, labelpad=10)
    plt.ylabel("Latitude", fontsize=28, labelpad=10)
    
    # rotated X-axis values
    plt.xticks(fontsize=20, rotation=35)
    plt.yticks(fontsize=20)

plt.tight_layout(pad=5)
plt.show()


# Range of eps values to test
eps_values = [0.05, 0.1, 0.15, 0.2]
min_samples = 5

sil_scores = []
cluster_counts = []

for eps in eps_values:
    # Create DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Count clusters (excluding noise)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    num_clusters = len(unique_labels)
    
    # Only calculate silhouette if more than 1 cluster
    if num_clusters > 1:
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        cluster_counts.append(num_clusters)
        print(f"eps = {eps}, clusters = {num_clusters}, score = {score:.3f}")
    else:
        print(f"eps = {eps}, clusters = {num_clusters} — too few to score")
        sil_scores.append(None)
        cluster_counts.append(num_clusters)

# Find best eps and score
best_idx = np.nanargmax([s if s is not None else -1 for s in sil_scores])
best_eps = eps_values[best_idx]
best_score = sil_scores[best_idx]

print(f"\nBest eps value: {best_eps}")
print(f"Best silhouette score: {best_score:.3f}")

# Visualization for DBSCAN Silhouette Scores
plt.figure(figsize=(10, 6))
plt.scatter(eps_values, [s if s is not None else 0 for s in sil_scores], s=80)
plt.plot(eps_values, [s if s is not None else 0 for s in sil_scores], linewidth=2)
plt.scatter([best_eps], [best_score], s=200, color='red', edgecolor='black')

# Move the text slightly lower and to the right
plt.text(best_eps, best_score - 0.05, 
         f"best eps = {best_eps}\nscore = {best_score:.3f}", 
         fontsize=14, va="top", ha="left", bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

plt.xlabel("eps (neighborhood size)", fontsize=28, labelpad=10)
plt.ylabel("Silhouette score", fontsize=28, labelpad=10)
plt.title("Silhouette score vs eps for DBSCAN", fontsize=18, pad=30)
plt.grid(True, alpha=0.4)
plt.xticks(fontsize=20, rotation=35)
plt.yticks(fontsize=20)
plt.tight_layout(pad=4)
plt.show()


# --- Comparison between KMeans and DBSCAN ---

# Fit KMeans with its best K
kmeans_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans_labels = kmeans_best.fit_predict(X_scaled)

# Fit DBSCAN with its best eps
dbscan_best = DBSCAN(eps=best_eps, min_samples=min_samples)
dbscan_labels = dbscan_best.fit_predict(X_scaled)

# Side-by-side visualization
plt.figure(figsize=(24, 10))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(X['Longitude'], X['Latitude'], c=kmeans_labels, s=20, cmap='tab10')
plt.title(f"K-Means (K={best_k})", fontsize=40, fontweight='bold', pad=20)
plt.xlabel("Longitude", fontsize=28, labelpad=10)
plt.ylabel("Latitude", fontsize=28, labelpad=10)
plt.xticks(fontsize=20, rotation=35)
plt.yticks(fontsize=20)
plt.grid(True, alpha=0.3)

# DBSCAN plot
plt.subplot(1, 2, 2)
plt.scatter(X['Longitude'], X['Latitude'], c=dbscan_labels, s=20, cmap='plasma')
plt.title(f"DBSCAN (eps={best_eps}, min_samples={min_samples})", fontsize=24, fontweight='bold', pad=20)
plt.xlabel("Longitude", fontsize=28, labelpad=10)
plt.ylabel("Latitude", fontsize=28, labelpad=10)
plt.xticks(fontsize=20, rotation=35)
plt.yticks(fontsize=20)
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=5)
plt.show()