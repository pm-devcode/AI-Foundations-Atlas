import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- The Function to Minimize ---
def func(x, y):
    return x**2 + 10 * y**2

# --- Training Loop ---
def train_torch(optimizer_class, start_params, lr, n_steps=50, **kwargs):
    # Parameters must be tensors with gradients enabled
    x = torch.tensor([start_params[0]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([start_params[1]], dtype=torch.float32, requires_grad=True)
    
    optimizer = optimizer_class([x, y], lr=lr, **kwargs)
    
    path = []
    path.append([x.item(), y.item()])
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = func(x, y)
        loss.backward()
        optimizer.step()
        
        path.append([x.item(), y.item()])
        
    return np.array(path)

# --- Execution ---
start_pos = [-8.0, 2.0]

# 1. SGD
path_sgd = train_torch(optim.SGD, start_pos, lr=0.05)

# 2. SGD with Momentum
path_momentum = train_torch(optim.SGD, start_pos, lr=0.01, momentum=0.9)

# 3. Adam
path_adam = train_torch(optim.Adam, start_pos, lr=0.5)

# --- Visualization ---
def plot_paths(paths, labels, colors, title, filename):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 10 * Y**2
    
    plt.figure(figsize=(10, 6))
    plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='gray', alpha=0.4)
    
    for path, label, color in zip(paths, labels, colors):
        plt.plot(path[:, 0], path[:, 1], 'o-', label=label, color=color, markersize=4, alpha=0.8)
        plt.plot(path[0, 0], path[0, 1], 'x', color=color)
        plt.plot(path[-1, 0], path[-1, 1], '*', color=color, markersize=10)
        
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

plot_paths([path_sgd, path_momentum, path_adam], 
           ['SGD', 'Momentum', 'Adam'], 
           ['red', 'blue', 'green'],
           "Optimization Algorithms Comparison (PyTorch)",
           "assets/pytorch_optimization.png")
