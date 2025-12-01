import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def sigmoid(z):
    """
    The Sigmoid activation function.
    Maps any real number to the (0, 1) interval.
    """
    return 1 / (1 + np.exp(-z))

def generate_data(num_points=100):
    """
    Generates two distinct clusters of data.
    Class 0: Centered at (2, 2)
    Class 1: Centered at (6, 6)
    """
    np.random.seed(42)
    
    # Class 0
    x0 = np.random.randn(num_points // 2, 2) + 2
    y0 = np.zeros((num_points // 2, 1))
    
    # Class 1
    x1 = np.random.randn(num_points // 2, 2) + 6
    y1 = np.ones((num_points // 2, 1))
    
    X = np.vstack((x0, x1))
    y = np.vstack((y0, y1))
    
    return X, y

def compute_cost(X, y, theta):
    """
    Computes the Log Loss (Binary Cross-Entropy).
    J(theta) = -(1/m) * sum(y*log(h) + (1-y)*log(1-h))
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # Epsilon to avoid log(0) error
    epsilon = 1e-5
    
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # 1. Prediction
        z = X.dot(theta)
        h = sigmoid(z)
        
        # 2. Error
        error = h - y
        
        # 3. Gradient
        gradient = (1/m) * X.T.dot(error)
        
        # 4. Update
        theta = theta - learning_rate * gradient
        
        # 5. Log Cost
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history

def main():
    # 1. Data
    X_raw, y = generate_data()
    
    # Add bias term
    X_b = np.c_[np.ones((len(X_raw), 1)), X_raw]
    
    # 2. Init
    theta = np.zeros((3, 1)) # 3 parameters: bias, w1, w2
    lr = 0.1
    iterations = 2000
    
    print("Starting Logistic Regression (Gradient Descent)...")
    print(f"Initial Cost: {compute_cost(X_b, y, theta):.4f}")
    
    # 3. Train
    theta_final, cost_history = gradient_descent(X_b, y, theta, lr, iterations)
    
    print(f"Final Cost: {cost_history[-1]:.4f}")
    print(f"Parameters: Bias={theta_final[0][0]:.2f}, W1={theta_final[1][0]:.2f}, W2={theta_final[2][0]:.2f}")
    
    # 4. Visualization
    
    # A. Decision Boundary
    plt.figure(figsize=(10, 6))
    
    # Plot points
    plt.scatter(X_raw[y.flatten()==0][:, 0], X_raw[y.flatten()==0][:, 1], color='blue', label='Class 0')
    plt.scatter(X_raw[y.flatten()==1][:, 0], X_raw[y.flatten()==1][:, 1], color='red', label='Class 1')
    
    # Plot Boundary Line
    # Decision boundary is where theta0 + theta1*x1 + theta2*x2 = 0
    # So x2 = -(theta0 + theta1*x1) / theta2
    x1_vals = np.array([np.min(X_raw[:, 0]), np.max(X_raw[:, 0])])
    x2_vals = -(theta_final[0] + theta_final[1] * x1_vals) / theta_final[2]
    
    plt.plot(x1_vals, x2_vals, "k--", linewidth=3, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig(os.path.join(ASSETS_DIR, 'decision_boundary.png'))
    print("Saved decision_boundary.png")
    
    # B. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), cost_history, color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Training Convergence')
    plt.savefig(os.path.join(ASSETS_DIR, 'loss_curve.png'))
    print("Saved loss_curve.png")
    
    # C. Sigmoid Visualization (Concept)
    z = np.linspace(-10, 10, 100)
    sig = sigmoid(z)
    plt.figure(figsize=(10, 4))
    plt.plot(z, sig, color='green')
    plt.axvline(0, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle=':')
    plt.title('The Sigmoid Function')
    plt.grid(True)
    plt.savefig(os.path.join(ASSETS_DIR, 'sigmoid_viz.png'))
    print("Saved sigmoid_viz.png")

if __name__ == "__main__":
    main()
