import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# Load Data
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# --- 1. PCA (Unsupervised) ---
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# --- 2. LDA (Supervised) ---
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# --- 3. Visualization Comparison ---
plt.figure(figsize=(14, 6))

# Plot PCA
plt.subplot(1, 2, 1)
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)

# Plot LDA
plt.subplot(1, 2, 2)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assets/lda_vs_pca.png")
print("Saved assets/lda_vs_pca.png")

# Print explained variance ratio for LDA
print('Explained variance ratio (LDA): %s'
      % str(lda.explained_variance_ratio_))
