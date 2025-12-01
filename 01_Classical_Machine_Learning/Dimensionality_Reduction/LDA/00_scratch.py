import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

class LDAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # 1. Compute the mean vector for each class
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            
            # 2. Compute Within-Class Scatter Matrix (S_W)
            # S_W = sum((x - mu_c)(x - mu_c)^T)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)
            
            # 3. Compute Between-Class Scatter Matrix (S_B)
            # S_B = sum(n_c * (mu_c - mu)(mu_c - mu)^T)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
            
        # 4. Solve Eigenvalue problem for S_W^-1 * S_B
        # We use np.linalg.eig on inv(S_W).dot(S_B)
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # 5. Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # 6. Store first n_components eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Project data
        return np.dot(X, self.linear_discriminants.T)

# --- Testing ---
# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Fit LDA
lda = LDAScratch(n_components=2)
lda.fit(X, y)
X_projected = lda.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

# Visualize
plt.figure(figsize=(10, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_projected[y == i, 0], X_projected[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA (Scratch) of IRIS dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.grid(True, alpha=0.3)
plt.savefig("assets/lda_scratch_result.png")
print("Saved assets/lda_scratch_result.png")
