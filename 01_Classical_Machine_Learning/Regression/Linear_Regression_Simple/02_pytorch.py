import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def generate_data(num_points=100, noise_factor=10):
    np.random.seed(42)
    X = 2 * np.random.rand(num_points, 1)
    y = 4 + 3 * X + np.random.randn(num_points, 1) * (noise_factor / 10.0)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    return X_tensor, y_tensor, X, y

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Linear layer: y = x * A + b
        self.linear = nn.Linear(1, 1) 

    def forward(self, x):
        return self.linear(x)

def main():
    # 1. Data Preparation
    X_tensor, y_tensor, X_numpy, y_numpy = generate_data()
    
    # 2. Model Definition
    model = LinearRegressionModel()
    
    # 3. Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    print("Starting PyTorch Training...")
    
    # 4. Training Loop
    epochs = 100
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad() # Clear gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights
        
        loss_history.append(loss.item())
    
    # 5. Results
    [w, b] = model.parameters()
    print(f"Final Loss: {loss_history[-1]:.4f}")
    print(f"Learned Parameters: Intercept={b.item():.2f}, Slope={w.item():.2f}")
    
    # 6. Visualization
    predicted = model(X_tensor).detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(X_numpy, y_numpy, color='blue', label='Data Points')
    plt.plot(X_numpy, predicted, color='purple', linewidth=2, label='PyTorch Prediction')
    plt.title('PyTorch Implementation')
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, 'pytorch_fit.png'))
    print("Saved pytorch_fit.png")

if __name__ == "__main__":
    main()
