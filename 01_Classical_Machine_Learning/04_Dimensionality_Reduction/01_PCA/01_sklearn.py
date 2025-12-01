import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Data Generation ---
X, y = make_blobs(n_samples=300, centers=1, random_state=42, cluster_std=1.0)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

# --- Sklearn Implementation ---
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

print(f"Sklearn Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Sklearn Components:\n{pca.components_}")

# --- Visualization ---
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='purple', edgecolor='k')
plt.title("Data Transformed by Sklearn PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.savefig('assets/sklearn_pca.png')
print("Saved assets/sklearn_pca.png")
