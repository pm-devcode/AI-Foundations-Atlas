import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

# --- 1. Synthetic Dataset (Circles) ---
def generate_data(num_samples=100, img_size=64):
    X = []
    y = []
    for _ in range(num_samples):
        img = np.zeros((img_size, img_size), dtype=np.float32)
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Random Circle
        cx, cy = np.random.randint(10, img_size-10, 2)
        radius = np.random.randint(5, 15)
        
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        mask_circle = dist_sq <= radius**2
        
        img[mask_circle] = 1.0
        mask[mask_circle] = 1.0
        
        # Add noise to image
        img += np.random.normal(0, 0.1, (img_size, img_size))
        
        X.append(img)
        y.append(mask)
        
    X = np.array(X).reshape(-1, 1, img_size, img_size)
    y = np.array(y).reshape(-1, 1, img_size, img_size)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 2. Simple U-Net Model ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32) # 64 because 32 (up) + 32 (skip)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16) # 32 because 16 (up) + 16 (skip)
        
        # Output
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u2 = self.up2(b)
        # Skip connection: concatenate u2 with e2
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        # Skip connection: concatenate u1 with e1
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        
        out = self.final(d1)
        return out

# --- 3. Training & Visualization ---
if __name__ == "__main__":
    # Generate Data
    X_train, y_train = generate_data(num_samples=200)
    X_test, y_test = generate_data(num_samples=5)
    
    model = SimpleUNet()
    criterion = nn.BCEWithLogitsLoss() # More stable than BCELoss + Sigmoid
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Simple U-Net...")
    for epoch in range(20): # Short training
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # Visualization
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test))
        preds = (preds > 0.5).float()
        
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i in range(3):
        # Input
        axes[i, 0].imshow(X_test[i, 0], cmap='gray')
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(y_test[i, 0], cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(preds[i, 0], cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis('off')
        
    plt.suptitle("U-Net Segmentation Results")
    
    output_path = os.path.join(assets_dir, "pytorch_unet_segmentation.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
