import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def generate_data(num_points=150):
    np.random.seed(42)
    x0 = np.random.randn(num_points // 3, 2) + [2, 2]
    y0 = np.zeros(num_points // 3)
    x1 = np.random.randn(num_points // 3, 2) + [6, 6]
    y1 = np.ones(num_points // 3)
    x2 = np.random.randn(num_points // 3, 2) + [2, 6]
    y2 = np.full(num_points // 3, 2)
    X = np.vstack((x0, x1, x2)).astype(np.float32)
    y = np.concatenate((y0, y1, y2)).astype(np.int32)
    return X, y

def knn_predict(X_train, y_train, X_test, k=5):
    """
    TensorFlow implementation of KNN prediction.
    """
    # Expand dims for broadcasting:
    # X_train: (num_train, 1, features)
    # X_test:  (1, num_test, features)
    # But to avoid OOM on large mesh, we loop or process in batches.
    # Here we process all at once since data is small.
    
    X_train_tf = tf.constant(X_train)
    X_test_tf = tf.constant(X_test)
    y_train_tf = tf.constant(y_train)
    
    # Calculate L2 distance
    # shape: (num_test, num_train)
    # We use squared difference -> sum -> sqrt
    # Broadcasting: (num_test, 1, 2) - (1, num_train, 2)
    diff = tf.expand_dims(X_test_tf, 1) - tf.expand_dims(X_train_tf, 0)
    dists = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))
    
    # Find top k (negative distance because top_k finds largest)
    values, indices = tf.math.top_k(-dists, k=k)
    
    # Gather labels
    nearest_labels = tf.gather(y_train_tf, indices)
    
    # Majority vote
    # We can't use tf.unique_with_counts easily on batch.
    # So we cast to numpy for the voting part or use a simple mode trick.
    # For simplicity in this demo, we'll just take the mode along axis 1.
    # Since TF doesn't have a direct 'mode', we'll do it in numpy.
    return nearest_labels.numpy()

def main():
    X, y = generate_data()
    k = 5
    
    # Visualization
    h = .1
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    X_query = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    
    print("TensorFlow: Computing distances...")
    nearest_labels = knn_predict(X, y, X_query, k=k)
    
    # Vote in numpy
    from scipy import stats
    Z, _ = stats.mode(nearest_labels, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(f"TF KNN (k = {k})")
    plt.savefig(os.path.join(ASSETS_DIR, 'tf_knn.png'))
    print("Saved tf_knn.png")

if __name__ == "__main__":
    main()
