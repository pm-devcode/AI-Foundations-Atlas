import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

class DBSCANScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1) # -1 represents Noise
        cluster_id = 0

        for i in range(n_samples):
            # If point is already visited (assigned to a cluster or noise), skip
            if self.labels[i] != -1:
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Mark as Noise (initially). Might be changed later if it's a border point.
                self.labels[i] = -1 
            else:
                # Found a core point, start a new cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
                
        return self.labels

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        # Assign the core point to the cluster
        self.labels[point_idx] = cluster_id
        
        # Process all neighbors (queue)
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If previously marked as Noise, change to Border Point (part of this cluster)
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
            
            # If not visited yet
            elif self.labels[neighbor_idx] == -1: # Wait, -1 is noise/unvisited. 
                # In this simple impl, we initialize everything to -1.
                # To distinguish "visited noise" from "unvisited", we could use a separate visited array.
                # But here, if it's -1, it means it hasn't been assigned to a valid cluster yet.
                pass

            # Actually, let's use a cleaner approach:
            # If it was unclassified (or noise), assign it.
            # If it was unclassified, check if it's ALSO a core point.
            
            # Let's refine the logic to match standard DBSCAN:
            # We need to know if a point was visited. 
            # But here, if label != -1, it's visited (either noise or cluster).
            # Wait, if I initialize to -1, I can't distinguish "Unvisited" from "Noise".
            # Let's use a separate visited set or array? 
            # Or just assume -1 is unvisited, and we might overwrite "Noise" with a cluster ID.
            pass
            
            i += 1

    # Let's rewrite fit/expand with a cleaner "visited" set logic for clarity
    pass

class DBSCANScratchRefined:
    def __init__(self, eps=0.3, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, 0) # 0: Unvisited
        # We will use: 0=Unvisited, -1=Noise, 1,2,3...=Cluster IDs
        
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != 0:
                continue
            
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1 # Noise
            else:
                cluster_id += 1
                self._expand_cluster(X, i, neighbors, cluster_id)
                
        # Adjust labels to match sklearn convention (-1 for noise, 0,1,2... for clusters)
        # Currently: -1 is noise, 1,2,3 are clusters.
        # Sklearn: -1 is noise, 0,1,2 are clusters.
        self.labels = np.where(self.labels > 0, self.labels - 1, self.labels)
        return self.labels

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if self.labels[neighbor_idx] == -1:
                # Was noise, now border point
                self.labels[neighbor_idx] = cluster_id
            
            elif self.labels[neighbor_idx] == 0:
                # Was unvisited
                self.labels[neighbor_idx] = cluster_id
                
                # Check if this neighbor is also a core point
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = neighbors + new_neighbors
            
            i += 1

    def _get_neighbors(self, X, point_idx):
        neighbors = []
        for i in range(X.shape[0]):
            # Euclidean distance
            dist = np.linalg.norm(X[point_idx] - X[i])
            if dist <= self.eps:
                neighbors.append(i)
        return neighbors

# --- Data Generation ---
# Moons dataset is perfect for DBSCAN (non-linear shapes)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# --- Training ---
# eps=0.2, min_samples=5 works well for this scale
dbscan = DBSCANScratchRefined(eps=0.2, min_samples=5)
labels = dbscan.fit(X)

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

plt.title(f"DBSCAN (Scratch) - Estimated clusters: {len(unique_labels) - (1 if -1 in labels else 0)}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("assets/scratch_dbscan.png")
print("Saved assets/scratch_dbscan.png")
