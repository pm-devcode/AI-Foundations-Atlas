import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def generate_data(num_points=100):
    np.random.seed(42)
    x0 = np.random.randn(num_points // 2, 2) + 2
    y0 = np.zeros((num_points // 2, 1))
    x1 = np.random.randn(num_points // 2, 2) + 6
    y1 = np.ones((num_points // 2, 1))
    X = np.vstack((x0, x1))
    y = np.vstack((y0, y1))
    
    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), X, y

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        # Sigmoid is applied in the loss function (BCEWithLogitsLoss) usually for stability,
        # but here we apply it explicitly to match the "scratch" logic or use BCELoss.
        # Let's use BCELoss, so we need sigmoid here.
        return torch.sigmoid(self.linear(x))

def main():
    X_tensor, y_tensor, X_numpy, y_numpy = generate_data()
    
    model = LogisticRegressionModel()
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    print("Starting PyTorch Training...")
    
    for epoch in range(1000):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Results
    final_loss = loss.item()
    with torch.no_grad():
        predicted = model(X_tensor)
        predicted_cls = (predicted > 0.5).float()
        accuracy = (predicted_cls == y_tensor).float().mean()
        
    print(f"Final Loss: {final_loss:.4f}, Accuracy: {accuracy*100:.1f}%")
    
    # Viz
    [w, b] = model.parameters()
    w = w.detach().numpy()[0]
    b = b.item()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_numpy[y_numpy.flatten()==0][:, 0], X_numpy[y_numpy.flatten()==0][:, 1], color='blue')
    plt.scatter(X_numpy[y_numpy.flatten()==1][:, 0], X_numpy[y_numpy.flatten()==1][:, 1], color='red')
    
    x1_vals = np.array([np.min(X_numpy[:, 0]), np.max(X_numpy[:, 0])])
    x2_vals = -(b + w[0] * x1_vals) / w[1]
    
    plt.plot(x1_vals, x2_vals, "k--", linewidth=3, label='PyTorch Boundary')
    plt.legend()
    plt.title('PyTorch Logistic Regression')
    plt.savefig(os.path.join(ASSETS_DIR, 'pytorch_boundary.png'))
    print("Saved pytorch_boundary.png")

if __name__ == "__main__":
    main()
