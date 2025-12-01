import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.tree = None

    def fit(self, X, current_height=0):
        """
        Recursively build the tree.
        """
        n_samples = X.shape[0]
        
        # Stop conditions:
        # 1. Reached height limit
        # 2. Only 1 sample left
        # 3. All samples are duplicates (cannot split)
        if current_height >= self.height_limit or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}
        
        # Check if all data points are identical
        if np.all(X == X[0]):
             return {"type": "leaf", "size": n_samples}

        # Randomly select a feature
        n_features = X.shape[1]
        feature_idx = random.randint(0, n_features - 1)
        
        # Randomly select a split value
        min_val = X[:, feature_idx].min()
        max_val = X[:, feature_idx].max()
        
        # If min == max, we can't split on this feature. Try finding another one or stop.
        if min_val == max_val:
             return {"type": "leaf", "size": n_samples}

        split_val = random.uniform(min_val, max_val)
        
        # Split data
        left_mask = X[:, feature_idx] < split_val
        right_mask = ~left_mask
        
        return {
            "type": "node",
            "feature_idx": feature_idx,
            "split_val": split_val,
            "left": self.fit(X[left_mask], current_height + 1),
            "right": self.fit(X[right_mask], current_height + 1)
        }

    def path_length(self, x, node, current_height=0):
        """
        Calculate path length for a single point x.
        """
        if node["type"] == "leaf":
            # Adjustment for unbuilt tree part (c(n))
            # If size > 1, we add an adjustment factor because the tree could have grown deeper
            if node["size"] > 1:
                return current_height + c(node["size"])
            return current_height
        
        if x[node["feature_idx"]] < node["split_val"]:
            return self.path_length(x, node["left"], current_height + 1)
        else:
            return self.path_length(x, node["right"], current_height + 1)

def c(n):
    """
    Average path length of unsuccessful search in BST.
    Harmonic number estimation.
    """
    if n <= 1:
        return 0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

class IsolationForestScratch:
    def __init__(self, n_trees=100, subsample_size=256):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.trees = []
        # Height limit is usually ceil(log2(subsample_size))
        self.height_limit = int(np.ceil(np.log2(subsample_size)))

    def fit(self, X):
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            # Subsampling
            indices = np.random.choice(n_samples, min(n_samples, self.subsample_size), replace=False)
            X_sub = X[indices]
            
            tree = IsolationTree(self.height_limit)
            root = tree.fit(X_sub)
            self.trees.append((tree, root))
            
    def anomaly_score(self, X):
        """
        Calculate anomaly score.
        s(x, n) = 2 ^ (- E(h(x)) / c(n) )
        """
        scores = []
        n_samples = self.subsample_size # Normalization factor uses subsample size
        
        for x in X:
            total_path_length = 0
            for tree, root in self.trees:
                total_path_length += tree.path_length(x, root)
            
            avg_path_length = total_path_length / self.n_trees
            score = 2 ** (-avg_path_length / c(n_samples))
            scores.append(score)
            
        return np.array(scores)

# --- Testing the Scratch Implementation ---

# 1. Generate Data
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2] # Two clusters
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)) # Random noise

X_all = np.r_[X_train, X_outliers]

# 2. Train
iso_forest = IsolationForestScratch(n_trees=100, subsample_size=64)
iso_forest.fit(X_train)

# 3. Score
scores = iso_forest.anomaly_score(X_all)

# 4. Visualize
plt.figure(figsize=(10, 6))
plt.title("Isolation Forest (Scratch) Anomaly Scores")

# Plot contour
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = iso_forest.anomaly_score(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# Plot points
# High score -> Anomaly (Darker blue in contour usually means higher score in this map? 
# Actually, let's check colorbar. Usually we want to highlight anomalies.)
# In this implementation:
# Short path -> High score (close to 1) -> Anomaly
# Long path -> Low score (close to 0) -> Normal

scatter = plt.scatter(X_all[:, 0], X_all[:, 1], c=scores, cmap='Reds', edgecolor='k', s=40)
plt.colorbar(scatter, label='Anomaly Score')
plt.legend(["Data points"], loc='upper right')
plt.savefig("assets/if_scratch_result.png")
print("Saved assets/if_scratch_result.png")
