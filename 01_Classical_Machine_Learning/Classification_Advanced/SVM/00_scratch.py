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

class LinearSVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Convert labels to {-1, 1} if they are {0, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Gradient only from regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Gradient from regularization + hinge loss
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# --- Data Generation & Visualization ---
if __name__ == "__main__":
    # Generate linearly separable data
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42, cluster_std=1.5)
    
    # Train Scratch Model
    svm = LinearSVMScratch(learning_rate=0.001, lambda_param=0.01, n_iters=2000)
    svm.fit(X, y)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm')
    
    # Plot the hyperplane and margins
    # w.x - b = 0  => x2 = (b - w1*x1) / w2
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)

    plt.plot([x0_1, x0_2], [x1_1, x1_2], 'k--', lw=2, label="Decision Boundary")
    plt.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k:', lw=1, label="Margin -1")
    plt.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k:', lw=1, label="Margin +1")
    
    plt.title("Linear SVM (Scratch Implementation - Gradient Descent)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    
    output_path = os.path.join(assets_dir, "scratch_svm_boundary.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
