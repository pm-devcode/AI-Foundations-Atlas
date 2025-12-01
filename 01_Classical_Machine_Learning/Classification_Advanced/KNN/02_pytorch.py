import torch
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
    y = np.concatenate((y0, y1, y2)).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y), X, y

def main():
    X_train, y_train, X_np, y_np = generate_data()
    k = 5
    
    # Visualization Grid
    h = .1
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    X_query = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    
    print("PyTorch: Computing distances...")
    
    # 1. Compute Distances (cdist is optimized for this)
    # dists shape: (num_query, num_train)
    dists = torch.cdist(X_query, X_train)
    
    # 2. Top K
    # largest=False means smallest distances
    values, indices = torch.topk(dists, k=k, largest=False)
    
    # 3. Gather Labels
    nearest_labels = y_train[indices] # shape (num_query, k)
    
    # 4. Vote
    # torch.mode returns (values, indices)
    predictions, _ = torch.mode(nearest_labels, dim=1)
    
    Z = predictions.numpy().reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(f"PyTorch KNN (k = {k})")
    plt.savefig(os.path.join(ASSETS_DIR, 'pytorch_knn.png'))
    print("Saved pytorch_knn.png")

if __name__ == "__main__":
    main()
