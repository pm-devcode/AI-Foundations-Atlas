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

class MLP_Multiclass_Scratch:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # Stability fix
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward(self, X, y_one_hot, output):
        m = X.shape[0]
        
        # Output Layer Error (dZ2 = A2 - Y) for Cross Entropy + Softmax
        dZ2 = output - y_one_hot
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden Layer Error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0) # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
    def fit(self, X, y, epochs=5000):
        # One-hot encode y
        n_values = np.max(y) + 1
        y_one_hot = np.eye(n_values)[y]
        
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y_one_hot, output)
            
            if i % 1000 == 0:
                loss = -np.mean(np.sum(y_one_hot * np.log(output + 1e-8), axis=1))
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# --- Data & Visualization ---
if __name__ == "__main__":
    # Generate 3 classes
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.5)
    
    # Train
    mlp = MLP_Multiclass_Scratch(input_size=2, hidden_size=10, output_size=3, learning_rate=0.1)
    mlp.fit(X, y, epochs=5000)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='viridis')
    plt.title("MLP Multiclass (Scratch Implementation)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    output_path = os.path.join(assets_dir, "scratch_mlp_multiclass.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
