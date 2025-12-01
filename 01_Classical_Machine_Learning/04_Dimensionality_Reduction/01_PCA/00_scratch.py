import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # 1. Standardize the data (Center it)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Compute Covariance Matrix
        # cov = (X^T * X) / (n - 1)
        cov_matrix = np.cov(X_centered.T)

        # 3. Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Sort Eigenvectors by Eigenvalues (descending)
        # eigenvectors[:, i] corresponds to eigenvalues[i]
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # 5. Store first n_components
        self.components = eigenvectors[:, :self.n_components]
        
        # Explained Variance Ratio
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        # Project data
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

# --- Data Generation ---
# Create correlated data
X, y = make_blobs(n_samples=300, centers=1, random_state=42, cluster_std=1.0)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

# --- Training ---
pca = PCAScratch(n_components=2)
pca.fit(X)
X_projected = pca.transform(X)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Components:\n{pca.components}")

# --- Visualization ---
plt.figure(figsize=(10, 5))

# Plot 1: Original Data with Principal Components
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c='blue', edgecolor='k')

mean = pca.mean
for length, vector in zip(pca.explained_variance_ratio_, pca.components.T):
    v = vector * 3 * np.sqrt(length) # Scale for visualization
    plt.arrow(mean[0], mean[1], v[0], v[1], color='red', width=0.05, head_width=0.15)

plt.title("Original Data & Principal Components")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Plot 2: Projected Data (1D if n_components=1, but here we kept 2 for rotation check)
# Let's project to 1D for the second plot to show dimensionality reduction
pca_1d = PCAScratch(n_components=1)
pca_1d.fit(X)
X_projected_1d = pca_1d.transform(X)

# To visualize 1D projection, we can plot it on a line
plt.subplot(1, 2, 2)
zeros = np.zeros_like(X_projected_1d)
plt.scatter(X_projected_1d, zeros, alpha=0.5, c='green', edgecolor='k')
plt.title("Projected Data (1D)")
plt.xlabel("Principal Component 1")
plt.yticks([])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/scratch_pca.png')
print("Saved assets/scratch_pca.png")
