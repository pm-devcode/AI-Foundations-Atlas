import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

# PyTorch MLP Model
class MulticlassMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MulticlassMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        # No Softmax here because nn.CrossEntropyLoss applies LogSoftmax internally

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

if __name__ == "__main__":
    # Generate Data
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.5)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) # Class indices for CrossEntropyLoss
    
    model = MulticlassMLP(input_size=2, hidden_size=10, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # Visualization
    plt.figure(figsize=(10, 6))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        logits = model(grid_tensor)
        _, predicted = torch.max(logits, 1)
        Z = predicted.numpy()
        
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='viridis')
    plt.title("PyTorch MLP Multiclass")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    output_path = os.path.join(assets_dir, "pytorch_mlp_multiclass.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
