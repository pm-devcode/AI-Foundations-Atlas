import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

class MLPScratch:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Initialize weights and biases
        # Weights between Input and Hidden
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Weights between Hidden and Output
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x) # Assuming x is already sigmoid output
    
    def forward(self, X):
        # Layer 1 (Hidden)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (Output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        # Output Layer Error
        # Loss derivative (MSE): (output - y)
        # Activation derivative: output * (1 - output)
        output_error = (output - y) * self.sigmoid_derivative(output)
        output_delta = output_error
        
        # Hidden Layer Error
        # Propagate error back: output_delta dot W2.T
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update Weights and Biases
        self.W2 -= self.lr * np.dot(self.a1.T, output_delta)
        self.b2 -= self.lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.W1 -= self.lr * np.dot(X.T, hidden_delta)
        self.b1 -= self.lr * np.sum(hidden_delta, axis=0, keepdims=True)
        
    def fit(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
    def predict(self, X):
        return self.forward(X)

# --- Data & Visualization ---
if __name__ == "__main__":
    # XOR Data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Train Scratch Model
    # 2 inputs -> 4 hidden neurons -> 1 output
    mlp = MLPScratch(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    mlp.fit(X, y, epochs=20000)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    
    # Plot decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=100, edgecolor='k', cmap='viridis')
    plt.title("MLP Scratch Implementation (XOR Problem)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    
    output_path = os.path.join(assets_dir, "scratch_mlp_xor.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
