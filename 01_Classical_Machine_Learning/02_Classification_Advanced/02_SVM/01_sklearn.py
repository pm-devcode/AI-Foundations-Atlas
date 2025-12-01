import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_moons

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Train Models
linear_svm = SVC(kernel='linear', C=1.0)
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')

linear_svm.fit(X, y)
rbf_svm.fit(X, y)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Helper function for plotting
def plot_boundary(ax, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plot_boundary(axes[0], linear_svm, "SVM Linear Kernel (Underfitting on Moons)")
plot_boundary(axes[1], rbf_svm, "SVM RBF Kernel (Handling Non-Linearity)")

output_path = os.path.join(assets_dir, "sklearn_svm_comparison.png")
plt.savefig(output_path)
print(f"Saved visualization to {output_path}")
