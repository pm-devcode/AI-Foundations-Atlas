import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

# --- Data & Visualization ---
if __name__ == "__main__":
    # Logic Gates Data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Labels
    y_and = np.array([0, 0, 0, 1])
    y_or  = np.array([0, 1, 1, 1])
    y_xor = np.array([0, 1, 1, 0])

    datasets = [("AND", y_and), ("OR", y_or), ("XOR", y_xor)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, y) in enumerate(datasets):
        p = Perceptron(learning_rate=0.1, n_iters=100)
        p.fit(X, y)
        
        ax = axes[i]
        
        # Plot decision boundary
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        Z = p.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
        ax.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolor='k', cmap='bwr')
        ax.set_title(f"Perceptron - {name} Gate")
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.grid(True, linestyle='--', alpha=0.6)

    output_path = os.path.join(assets_dir, "perceptron_gates.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
