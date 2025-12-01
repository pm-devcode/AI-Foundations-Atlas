import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent window popping up
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_data(num_points=100, noise_factor=10):
    """
    Generates synthetic linear data with noise.
    y = 3x + 4 + noise
    """
    np.random.seed(42)
    X = 2 * np.random.rand(num_points, 1)
    y = 4 + 3 * X + np.random.randn(num_points, 1) * (noise_factor / 10.0)
    return X, y

def compute_cost(X, y, theta):
    """
    Computes the Mean Squared Error (MSE) cost function.
    J(theta) = (1/2m) * sum((h_theta(x) - y)^2)
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Performs Gradient Descent to optimize theta.
    Returns the optimized theta and the history of cost values.
    """
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # 1. Calculate Predictions: h_theta(x) = X * theta
        predictions = X.dot(theta)
        
        # 2. Calculate Errors: (h_theta(x) - y)
        errors = predictions - y
        
        # 3. Calculate Gradients: (1/m) * X.T * errors
        # This corresponds to the partial derivatives derived in README.md
        gradients = (1 / m) * X.T.dot(errors)
        
        # 4. Update Parameters: theta = theta - alpha * gradients
        theta = theta - learning_rate * gradients
        
        # 5. Record Cost
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history

def main():
    # 1. Data Preparation
    X_raw, y = generate_data()
    
    # Add bias term (x0 = 1) to X
    # X_b becomes [1, x1; 1, x2; ...]
    X_b = np.c_[np.ones((len(X_raw), 1)), X_raw]
    
    # 2. Initialization
    # Random initialization of theta (weights)
    theta = np.random.randn(2, 1)
    
    # Hyperparameters
    learning_rate = 0.1
    iterations = 1000
    
    print("Starting Gradient Descent...")
    print(f"Initial Cost: {compute_cost(X_b, y, theta):.4f}")
    
    # 3. Training
    theta_final, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)
    
    print(f"Final Cost: {cost_history[-1]:.4f}")
    print(f"Learned Parameters: Intercept={theta_final[0][0]:.2f}, Slope={theta_final[1][0]:.2f}")
    print(f"True Parameters: Intercept=4.00, Slope=3.00")

    # 4. Visualization
    
    # Plot A: Regression Line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_raw, y, color='blue', label='Data Points')
    plt.plot(X_raw, X_b.dot(theta_final), color='red', linewidth=2, label='Prediction (Regression Line)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ASSETS_DIR, 'regression_fit.png'))
    print("Saved regression_fit.png")
    
    # Plot B: Cost History
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), cost_history, color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Cost J(theta)')
    plt.title('Convergence of Gradient Descent')
    plt.grid(True)
    plt.savefig(os.path.join(ASSETS_DIR, 'loss_curve.png'))
    print("Saved loss_curve.png")

if __name__ == "__main__":
    main()
