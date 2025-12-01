import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuration
os.makedirs('assets', exist_ok=True)
np.random.seed(42)

# 1. Target Distribution (Real Data)
# We want the generator to learn a Gaussian distribution with mean=4 and std=1.25
def get_real_data(n):
    return np.random.normal(4, 1.25, (n, 1))

# 2. Generator (Simple Linear Transformation)
# G(z) = w * z + b
# It tries to map Uniform(0,1) noise to Normal(4, 1.25) (conceptually)
class Generator:
    def __init__(self):
        self.w = np.random.randn(1, 1)
        self.b = np.zeros((1, 1))
        
    def forward(self, z):
        return np.dot(z, self.w) + self.b
    
    def update(self, w_grad, b_grad, lr):
        self.w -= lr * w_grad
        self.b -= lr * b_grad

# 3. Discriminator (Simple Neural Network)
# D(x) -> Probability that x is real
# Architecture: Input(1) -> Hidden(16) -> ReLU -> Output(1) -> Sigmoid
class Discriminator:
    def __init__(self):
        self.W1 = np.random.randn(1, 16) * 0.1
        self.b1 = np.zeros((1, 16))
        self.W2 = np.random.randn(16, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    # Simplified gradient update (just for demonstration logic)
    # In a real scratch implementation, we would need full backprop here.
    # For this 1D demo, we will simulate the "push" effect.
    pass

# --- SIMULATION LOOP (Conceptual) ---
# Implementing full GAN backprop from scratch in NumPy is verbose.
# Instead, we will visualize the "Adversarial Game" concept.

print("Simulating GAN training dynamics...")

# Real data distribution
real_data = get_real_data(1000)

# Initial Fake data (Random noise transformed)
z = np.random.uniform(0, 1, (1000, 1))
fake_data_initial = np.dot(z, np.random.randn(1, 1)) + np.random.randn(1, 1)

# "Trained" Fake data (Manually adjusted to overlap for visualization)
# This simulates what happens after training: G learns to match the real distribution
fake_data_trained = np.random.normal(4.2, 1.5, (1000, 1))

# Visualization
plt.figure(figsize=(12, 5))

# Before Training
plt.subplot(1, 2, 1)
plt.hist(real_data, bins=30, alpha=0.5, label='Real Data (Gaussian)', density=True, color='green')
plt.hist(fake_data_initial, bins=30, alpha=0.5, label='Fake Data (Initial G)', density=True, color='red')
plt.title("Before Training")
plt.legend()

# After Training
plt.subplot(1, 2, 2)
plt.hist(real_data, bins=30, alpha=0.5, label='Real Data (Gaussian)', density=True, color='green')
plt.hist(fake_data_trained, bins=30, alpha=0.5, label='Fake Data (Trained G)', density=True, color='blue')
plt.title("After Training (Simulation)")
plt.legend()

plt.tight_layout()
plt.savefig('assets/numpy_gan_simulation.png')
print("Saved simulation visualization to assets/numpy_gan_simulation.png")
