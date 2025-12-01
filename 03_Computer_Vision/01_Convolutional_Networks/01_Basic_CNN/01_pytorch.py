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

# --- 1. Synthetic Dataset Generation ---
def generate_dataset(num_samples=200, img_size=16):
    X = []
    y = []
    for _ in range(num_samples):
        img = np.zeros((img_size, img_size), dtype=np.float32)
        label = np.random.randint(0, 2) # 0: Vertical, 1: Horizontal
        
        if label == 0: # Vertical Line
            col = np.random.randint(2, img_size-2)
            img[2:-2, col] = 1.0
            # Add noise
            img += np.random.normal(0, 0.1, (img_size, img_size))
        else: # Horizontal Line
            row = np.random.randint(2, img_size-2)
            img[row, 2:-2] = 1.0
            # Add noise
            img += np.random.normal(0, 0.1, (img_size, img_size))
            
        X.append(img)
        y.append(label)
        
    X = np.array(X).reshape(-1, 1, img_size, img_size) # (N, Channels, H, W)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- 2. Simple CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input channel (grayscale), 4 output channels (filters), 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(4 * 8 * 8, 2) # 16/2 = 8 spatial dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# --- 3. Training & Visualization ---
if __name__ == "__main__":
    # Generate Data
    X, y = generate_dataset()
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Simple CNN...")
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Visualize Learned Filters
    filters = model.conv1.weight.data.numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    # Plot Loss
    axes[0].plot(losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    
    # Plot Filters
    for i in range(4):
        ax = axes[i+1]
        ax.imshow(filters[i, 0], cmap='gray')
        ax.set_title(f"Learned Filter {i+1}")
        ax.axis('off')
        
    plt.suptitle("PyTorch CNN: Training on Vertical vs Horizontal Lines")
    
    output_path = os.path.join(assets_dir, "pytorch_cnn_filters.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
