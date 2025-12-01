import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Data Generation ---
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# --- Sklearn Implementation ---
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

# --- Visualization ---
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title(f"Sklearn DBSCAN - Estimated clusters: {len(unique_labels) - (1 if -1 in labels else 0)}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("assets/sklearn_dbscan.png")
print("Saved assets/sklearn_dbscan.png")
