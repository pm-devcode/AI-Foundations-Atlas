import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

# Generate same data
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Train Sklearn Model
clf = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
clf.fit(X, y)

# Visualization
plt.figure(figsize=(10, 6))

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='viridis')
plt.title("Decision Tree (Scikit-Learn Reference)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

output_path = os.path.join(assets_dir, "sklearn_tree_boundary.png")
plt.savefig(output_path)
print(f"Saved visualization to {output_path}")