import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

# Single Neuron Model
class PyTorchPerceptron(nn.Module):
    def __init__(self):
        super(PyTorchPerceptron, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.activation = nn.Sigmoid() # Using Sigmoid for smooth gradient descent

    def forward(self, x):
        return self.activation(self.linear(x))

def train_and_plot(X, y, ax, title):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    model = PyTorchPerceptron()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Training
    for _ in range(1000):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
    # Visualization
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolor='k', cmap='bwr')
    ax.set_title(f"PyTorch Neuron - {title}")
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")

if __name__ == "__main__":
    # Logic Gates Data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    y_and = np.array([0, 0, 0, 1])
    y_or  = np.array([0, 1, 1, 1])
    y_xor = np.array([0, 1, 1, 0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    train_and_plot(X, y_and, axes[0], "AND")
    train_and_plot(X, y_or, axes[1], "OR")
    train_and_plot(X, y_xor, axes[2], "XOR")
    
    output_path = os.path.join(assets_dir, "pytorch_neuron_gates.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
