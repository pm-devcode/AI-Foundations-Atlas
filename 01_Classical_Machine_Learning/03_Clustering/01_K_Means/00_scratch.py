import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_data(num_points=300, k=3):
    """
    Generates synthetic blobs of data.
    """
    np.random.seed(42)
    
    # Centers for the blobs
    centers = [[2, 2], [8, 3], [5, 9]]
    X = []
    
    for c in centers:
        # Generate points around the center with some noise
        points = np.random.randn(num_points // k, 2) + c
        X.append(points)
        
    X = np.vstack(X)
    return X

def compute_distances(X, centroids):
    """
    Computes Euclidean distance from each point to each centroid.
    Returns matrix of shape (m, K)
    """
    m = X.shape[0]
    K = centroids.shape[0]
    distances = np.zeros((m, K))
    
    for k in range(K):
        # || x - mu_k ||^2
        # We use norm along axis 1
        norm = np.linalg.norm(X - centroids[k], axis=1)
        distances[:, k] = norm
        
    return distances

def k_means(X, K, max_iters=100):
    m, n = X.shape
    
    # 1. Initialize Centroids (Randomly pick K points from X)
    indices = np.random.choice(m, K, replace=False)
    centroids = X[indices]
    
    previous_centroids = centroids.copy()
    
    for i in range(max_iters):
        # 2. Expectation (Assign points to nearest centroid)
        distances = compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        
        # 3. Maximization (Update centroids)
        for k in range(K):
            # Get all points assigned to cluster k
            points_k = X[labels == k]
            
            if len(points_k) > 0:
                centroids[k] = np.mean(points_k, axis=0)
                
        # 4. Check Convergence
        shift = np.linalg.norm(centroids - previous_centroids)
        if shift < 1e-4:
            print(f"Converged at iteration {i}")
            break
            
        previous_centroids = centroids.copy()
        
    return centroids, labels

def main():
    # 1. Data
    X = generate_data()
    K = 3
    
    print("Starting K-Means (Scratch)...")
    
    # 2. Train
    centroids, labels = k_means(X, K)
    
    # 3. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot points colored by label
    colors = ['r', 'g', 'b']
    for k in range(K):
        cluster_points = X[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k}', alpha=0.6)
        
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.title('K-Means Clustering (Scratch)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(ASSETS_DIR, 'kmeans_clusters.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
