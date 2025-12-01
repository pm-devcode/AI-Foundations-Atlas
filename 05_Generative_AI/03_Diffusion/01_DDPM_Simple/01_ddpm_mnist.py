import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import math

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 3
TIMESTEPS = 300
IMAGE_SIZE = 28
CHANNELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create assets directory
os.makedirs("assets", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Data Loading
# -----------------------------------------------------------------------------
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# -----------------------------------------------------------------------------
# 2. Diffusion Utilities
# -----------------------------------------------------------------------------
class Diffusion:
    def __init__(self, timesteps=TIMESTEPS, beta_start=0.0001, beta_end=0.02, device=DEVICE):
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=self.device)

# -----------------------------------------------------------------------------
# 3. Model Architecture (Corrected U-Net)
# -----------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = CHANNELS
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i], time_emb_dim) \
            for i in range(len(down_channels))
        ])
        
        self.down_sample = nn.ModuleList([
            nn.Conv2d(down_channels[i], down_channels[i+1], 4, 2, 1) \
            for i in range(len(down_channels)-1)
        ])
        
        self.ups = nn.ModuleList([
            Block(up_channels[i] + down_channels[-(i+2)], up_channels[i+1], time_emb_dim, up=True) \
            for i in range(len(up_channels)-1)
        ])
        
        self.up_sample = nn.ModuleList([
            nn.ConvTranspose2d(up_channels[i], up_channels[i], 4, 2, 1) \
            for i in range(len(up_channels)-1)
        ])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        residuals = []
        
        for i, down in enumerate(self.downs):
            x = down(x, t)
            if i < len(self.downs) - 1:
                residuals.append(x)
                x = self.down_sample[i](x)
            else:
                pass
                
        for i, up in enumerate(self.ups):
            x = self.up_sample[i](x)
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
            
        return self.output(x)

# -----------------------------------------------------------------------------
# 4. Training & Sampling Logic
# -----------------------------------------------------------------------------
def train():
    dataloader = get_dataloader()
    model = SimpleUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    diffusion = Diffusion(device=DEVICE)
    loss_fn = nn.MSELoss()
    
    print(f"Starting DDPM Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.forward_diffusion_sample(images, t)
            
            predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return model, diffusion

@torch.no_grad()
def sample(model, diffusion):
    print("Sampling from the model...")
    model.eval()
    
    x = torch.randn((16, 1, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    
    for i in reversed(range(1, diffusion.timesteps)):
        t = (torch.ones(16) * i).long().to(DEVICE)
        predicted_noise = model(x, t)
        
        alpha = diffusion.alphas[i]
        alpha_hat = diffusion.alphas_cumprod[i]
        beta = diffusion.betas[i]
        
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

def save_images(images, path):
    grid = torchvision.utils.make_grid(images, nrow=4)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.figure(figsize=(5,5))
    plt.imshow(ndarr, cmap='gray')
    plt.axis('off')
    plt.savefig(path)
    plt.close()
    print(f"Saved generated images to {path}")

if __name__ == "__main__":
    model, diffusion = train()
    generated_images = sample(model, diffusion)
    save_images(generated_images, "assets/ddpm_generated.png")
