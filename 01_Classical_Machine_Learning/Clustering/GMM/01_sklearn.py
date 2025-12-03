import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Data Generation ---
# Generate anisotropicly distributed data (stretched blobs)
# GMM shines where K-Means fails (non-spherical clusters)
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

# --- 2. Scikit-Learn GMM ---
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_stretched)
y_gmm = gmm.predict(X_stretched)

# --- 3. Visualization ---
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=y_gmm, s=20, cmap='viridis', zorder=2)

# Plot ellipses
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle=angle, **kwargs))

ax = plt.gca()
for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    draw_ellipse(pos, covar, alpha=0.2 * w)

plt.title("GMM Clustering (Scikit-Learn) on Anisotropic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)
plt.savefig("assets/gmm_sklearn_result.png")
print("Saved assets/gmm_sklearn_result.png")

# --- 4. AIC/BIC for Model Selection ---
# How to choose K?
n_components = np.arange(1, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X_stretched)
          for n in n_components]

plt.figure(figsize=(8, 5))
plt.plot(n_components, [m.bic(X_stretched) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_stretched) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.ylabel('Information Criterion')
plt.title("Model Selection (AIC/BIC)")
plt.grid(True)
plt.savefig("assets/gmm_model_selection.png")
print("Saved assets/gmm_model_selection.png")
