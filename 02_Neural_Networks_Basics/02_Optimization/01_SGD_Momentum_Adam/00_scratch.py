import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- The Function to Minimize ---
# f(x, y) = x^2 + 10y^2
# This is an elongated bowl. Gradients are much steeper in y than x.
def func(x, y):
    return x**2 + 10 * y**2

def grad_func(x, y):
    # df/dx = 2x
    # df/dy = 20y
    return np.array([2 * x, 20 * y])

# --- Optimizers ---

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def step(self, params, grads):
        return params - self.lr * grads

class Momentum:
    def __init__(self, lr=0.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = np.zeros(2) # Velocity
        
    def step(self, params, grads):
        self.v = self.gamma * self.v + self.lr * grads
        return params - self.v

class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(2) # First moment
        self.v = np.zeros(2) # Second moment
        self.t = 0
        
    def step(self, params, grads):
        self.t += 1
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# --- Training Loop ---
def train(optimizer, start_params, n_steps=50):
    params = np.array(start_params, dtype=float)
    path = [params.copy()]
    
    for _ in range(n_steps):
        grads = grad_func(params[0], params[1])
        params = optimizer.step(params, grads)
        path.append(params.copy())
        
    return np.array(path)

# --- Execution ---
start_pos = [-8.0, 2.0] # Start far away

# 1. SGD
# Needs small LR to not diverge on the steep y-axis, but then it's slow on x-axis
sgd = SGD(lr=0.05) 
path_sgd = train(sgd, start_pos)

# 2. Momentum
# Can handle larger LR and gains speed
momentum = Momentum(lr=0.01, gamma=0.9)
path_momentum = train(momentum, start_pos)

# 3. Adam
# Adaptive LR handles the different scales of x and y gradients well
adam = Adam(lr=0.5)
path_adam = train(adam, start_pos)

# --- Visualization ---
def plot_paths(paths, labels, colors, title, filename):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    plt.figure(figsize=(10, 6))
    plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='gray', alpha=0.4)
    
    for path, label, color in zip(paths, labels, colors):
        plt.plot(path[:, 0], path[:, 1], 'o-', label=label, color=color, markersize=4, alpha=0.8)
        # Mark start and end
        plt.plot(path[0, 0], path[0, 1], 'x', color=color) # Start
        plt.plot(path[-1, 0], path[-1, 1], '*', color=color, markersize=10) # End
        
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
           "Optimization Algorithms Comparison (Scratch)",
           "assets/scratch_optimization.png")
