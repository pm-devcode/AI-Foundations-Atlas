import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function: Reconstruction + KL Divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# --- Training Loop (Simplified for Demo) ---
def train():
    # Hyperparameters
    batch_size = 128
    epochs = 5 # Short run for demo
    lr = 1e-3
    
    # Data
    # We won't actually download MNIST to avoid network issues/time, 
    # but this is how it would be set up.
    # We will use random noise to verify the architecture runs.
    
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training simulation with random data...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Simulate a batch
        data = torch.rand(batch_size, 1, 28, 28) 
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        print(f'Epoch: {epoch+1}, Loss: {train_loss / batch_size:.4f}')

    # Generate sample
    with torch.no_grad():
        z = torch.randn(1, 20)
        sample = model.decode(z).view(28, 28)
        plt.imshow(sample.numpy(), cmap='gray')
        plt.title("VAE Generated Sample (Random Noise Training)")
        plt.savefig("assets/vae_sample.png")
        print("Saved assets/vae_sample.png")

if __name__ == "__main__":
    train()
