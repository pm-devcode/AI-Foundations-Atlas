import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from collections import Counter

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_data(num_points=150):
    """
    Generates 3 clusters of data.
    """
    np.random.seed(42)
    
    # Cluster 0
    x0 = np.random.randn(num_points // 3, 2) + [2, 2]
    y0 = np.zeros(num_points // 3)
    
    # Cluster 1
    x1 = np.random.randn(num_points // 3, 2) + [6, 6]
    y1 = np.ones(num_points // 3)
    
    # Cluster 2
    x2 = np.random.randn(num_points // 3, 2) + [2, 6]
    y2 = np.full(num_points // 3, 2)
    
    X = np.vstack((x0, x1, x2))
    y = np.concatenate((y0, y1, y2))
    
    return X, y

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Lazy learning: just store the data.
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
        
    def _predict_single(self, x):
        # 1. Compute distances
        # Euclidean distance: sqrt(sum((x1-x2)^2))
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # 2. Get K nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 3. Majority Vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def main():
    # 1. Data
    X, y = generate_data()
    
    # 2. Model
    k = 5
    clf = KNN(k=k)
    clf.fit(X, y)
    
    print(f"KNN (K={k}) initialized.")
    
    # 3. Visualization (Decision Boundary)
    h = .1  # step size in the mesh
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Calculate min, max and limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    print("Generating decision boundary (this might take a moment)...")
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"3-Class classification (k = {k})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    save_path = os.path.join(ASSETS_DIR, 'knn_boundary.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
