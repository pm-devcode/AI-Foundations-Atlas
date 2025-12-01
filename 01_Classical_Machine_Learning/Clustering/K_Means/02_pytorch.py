import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def generate_data(num_points=300, k=3):
    np.random.seed(42)
    centers = [[2, 2], [8, 3], [5, 9]]
    X = []
    for c in centers:
        points = np.random.randn(num_points // k, 2) + c
        X.append(points)
    X = np.vstack(X).astype(np.float32)
    return torch.from_numpy(X)

def main():
    X = generate_data()
    K = 3
    m = X.shape[0]
    
    # 1. Initialize Centroids
    indices = torch.randperm(m)[:K]
    centroids = X[indices]
    
    print("Starting PyTorch K-Means...")
    
    for i in range(100):
        # 2. Expectation
        # cdist computes pairwise euclidean distance
        distances = torch.cdist(X, centroids) # (m, K)
        labels = torch.argmin(distances, dim=1) # (m,)
        
        # 3. Maximization
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            points_k = X[labels == k]
            if len(points_k) > 0:
                new_centroids[k] = points_k.mean(dim=0)
            else:
                # Handle empty cluster (keep old centroid or re-init)
                new_centroids[k] = centroids[k]
                
        # Check convergence
        shift = torch.norm(new_centroids - centroids)
        if shift < 1e-4:
            print(f"Converged at iteration {i}")
            centroids = new_centroids
            break
            
        centroids = new_centroids
        
    # Visualization
    X_np = X.numpy()
    centroids_np = centroids.numpy()
    labels_np = labels.numpy()
    
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    for k in range(K):
        cluster_points = X_np[labels_np == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], alpha=0.6)
        
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1], c='black', marker='x', s=200, linewidths=3)
    plt.title('PyTorch K-Means')
    plt.savefig(os.path.join(ASSETS_DIR, 'pytorch_kmeans.png'))
    print("Saved pytorch_kmeans.png")

if __name__ == "__main__":
    main()
