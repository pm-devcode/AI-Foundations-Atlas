import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Configuration
os.makedirs('assets', exist_ok=True)
torch.manual_seed(42)

# 1. Data
digits = load_digits()
X = digits.data / 16.0 # Normalization 0-1
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long) # For coloring the plot

# 2. Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim) # Bottleneck
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Output 0-1 (since data is normalized)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

input_dim = 64 # 8x8
latent_dim = 2 # Compression to 2D for visualization
model = Autoencoder(input_dim, latent_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. Training
EPOCHS = 500
print("Starting Autoencoder training (PyTorch)...")

losses = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs, latent = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. Evaluation and Visualization
model.eval()
with torch.no_grad():
    test_outputs, test_latent = model(X_test_tensor)

# A. Latent Space
plt.figure(figsize=(8, 6))
z = test_latent.numpy()
plt.scatter(z[:, 0], z[:, 1], c=y_test, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(label='Digit Label')
plt.title("Latent Space (Non-linear Autoencoder)")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.grid(True, alpha=0.3)
plt.savefig('assets/pytorch_latent_space.png')
print("Saved latent space visualization to assets/pytorch_latent_space.png")

# B. Reconstruction
n_samples = 5
indices = torch.randperm(len(X_test_tensor))[:n_samples]

plt.figure(figsize=(10, 4))
for i, idx in enumerate(indices):
    # Original
    ax = plt.subplot(2, n_samples, i + 1)
    plt.imshow(X_test_tensor[idx].reshape(8, 8), cmap='gray')
    plt.title(f"Org: {y_test[idx]}")
    plt.axis('off')
    
    # Reconstruction
    ax = plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(test_outputs[idx].reshape(8, 8), cmap='gray')
    plt.title("Rec")
    plt.axis('off')

plt.suptitle(f"Reconstruction (Non-linear Compression)")
plt.tight_layout()
plt.savefig('assets/pytorch_reconstruction.png')
print("Saved reconstruction visualization to assets/pytorch_reconstruction.png")
