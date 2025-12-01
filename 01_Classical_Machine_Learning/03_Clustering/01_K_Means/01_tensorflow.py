import tensorflow as tf
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
    return X

def main():
    X_np = generate_data()
    X = tf.constant(X_np)
    K = 3
    m = X.shape[0]
    
    # 1. Initialize Centroids
    # Randomly select K indices
    indices = tf.random.shuffle(tf.range(m))[:K]
    centroids = tf.gather(X, indices)
    
    print("Starting TensorFlow K-Means...")
    
    for i in range(100):
        # 2. Expectation (Assign points)
        # Expand dims for broadcasting: (m, 1, 2) - (1, K, 2)
        diff = tf.expand_dims(X, 1) - tf.expand_dims(centroids, 0)
        distances = tf.reduce_sum(tf.square(diff), axis=2) # (m, K)
        labels = tf.argmin(distances, axis=1) # (m,)
        
        # 3. Maximization (Update centroids)
        # We use unsorted_segment_mean to calculate mean of points for each label
        new_centroids = tf.math.unsorted_segment_mean(X, labels, num_segments=K)
        
        # Check convergence
        shift = tf.reduce_sum(tf.square(new_centroids - centroids))
        if shift < 1e-4:
            print(f"Converged at iteration {i}")
            centroids = new_centroids
            break
            
        centroids = new_centroids
        
    # Visualization
    centroids_np = centroids.numpy()
    labels_np = labels.numpy()
    
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    for k in range(K):
        cluster_points = X_np[labels_np == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], alpha=0.6)
        
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1], c='black', marker='x', s=200, linewidths=3)
    plt.title('TensorFlow K-Means')
    plt.savefig(os.path.join(ASSETS_DIR, 'tf_kmeans.png'))
    print("Saved tf_kmeans.png")

if __name__ == "__main__":
    main()
