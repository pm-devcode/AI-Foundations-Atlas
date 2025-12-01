import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from sklearn.datasets import load_digits

# Configuration
os.makedirs('assets', exist_ok=True)
np.random.seed(42)

# 1. Data (Digits - 8x8 images)
digits = load_digits()
X = digits.data # (1797, 64)
# Normalization to 0-1
X = X / 16.0

print(f"Input data shape: {X.shape}")

# 2. Mathematical "Autoencoder" (SVD / PCA)
# A linear autoencoder converges to the PCA solution.
# SVD Decomposition: X = U * S * Vt
# X: [N, D]
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# 3. Dimensionality Reduction (Encoder)
# Keep only the k most important features
k = 2 # Latent dimension
Z = np.dot(X, Vt[:k].T) # Projection onto k principal components
print(f"Latent space shape: {Z.shape}")

# 4. Reconstruction (Decoder)
# X_hat = Z * V_k
X_reconstructed = np.dot(Z, Vt[:k])
print(f"Reconstructed shape: {X_reconstructed.shape}")

# 5. Latent Space Visualization
plt.figure(figsize=(8, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=digits.target, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(label='Digit Label')
plt.title("Latent Space (SVD/Linear Autoencoder)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True, alpha=0.3)
plt.savefig('assets/numpy_latent_space.png')
print("Saved latent space visualization to assets/numpy_latent_space.png")

# 6. Reconstruction Visualization
n_samples = 5
indices = np.random.choice(len(X), n_samples, replace=False)

plt.figure(figsize=(10, 4))
for i, idx in enumerate(indices):
    # Original
    ax = plt.subplot(2, n_samples, i + 1)
    plt.imshow(X[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Org: {digits.target[idx]}")
    plt.axis('off')
    
    # Reconstruction
    ax = plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(X_reconstructed[idx].reshape(8, 8), cmap='gray')
    plt.title("Rec")
    plt.axis('off')

plt.suptitle(f"Reconstruction with {k} components (Linear Compression)")
plt.tight_layout()
plt.savefig('assets/numpy_reconstruction.png')
print("Saved reconstruction visualization to assets/numpy_reconstruction.png")
