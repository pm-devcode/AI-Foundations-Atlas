import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Data Preparation (Zachary's Karate Club) ---
G = nx.karate_club_graph()
num_nodes = G.number_of_nodes()

# Get labels (Club: 'Mr. Hi' vs 'Officer')
labels = []
for i in range(num_nodes):
    club = G.nodes[i]['club']
    labels.append(0 if club == 'Mr. Hi' else 1)
labels = torch.tensor(labels, dtype=torch.long)

# Adjacency Matrix
A = nx.to_numpy_array(G)
A = torch.tensor(A, dtype=torch.float32)

# Features: Identity Matrix (One-hot encoding of node ID)
# In real tasks, this would be text features, etc.
X = torch.eye(num_nodes, dtype=torch.float32)

# --- 2. GCN Layer (PyTorch) ---
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, A_hat, X):
        # A_hat: Normalized Adjacency Matrix
        # X: Node Features
        
        # 1. Aggregation: A_hat * X
        agg = torch.mm(A_hat, X)
        
        # 2. Transformation: agg * W
        out = self.linear(agg)
        
        return out

class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(num_features, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, A_hat, X):
        h = self.gcn1(A_hat, X)
        h = self.relu(h)
        out = self.gcn2(A_hat, h)
        return out, h

# --- 3. Preprocessing (Normalization) ---
# A_hat = D^-0.5 * (A + I) * D^-0.5
I = torch.eye(num_nodes)
A_tilde = A + I
D_tilde = torch.sum(A_tilde, dim=1)
D_inv_sqrt = torch.diag(torch.pow(D_tilde, -0.5))
A_hat = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)

# --- 4. Training ---
model = GCN(num_features=num_nodes, hidden_size=16, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Semi-Supervised: Only train on Node 0 (Mr. Hi) and Node 33 (Officer)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[0] = True
train_mask[33] = True

losses = []
print("Training GCN on Karate Club...")

for epoch in range(200):
    optimizer.zero_grad()
    output, hidden = model(A_hat, X)
    
    # Only calculate loss on labeled nodes
    loss = criterion(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --- 5. Visualization ---
model.eval()
_, embeddings = model(A_hat, X)
embeddings = embeddings.detach().numpy()

plt.figure(figsize=(12, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")

# Plot 2: Embeddings
plt.subplot(1, 2, 2)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='coolwarm', s=100, edgecolors='k')
plt.title("Node Embeddings (Hidden Layer)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

# Annotate key nodes
plt.text(embeddings[0, 0], embeddings[0, 1], "Mr. Hi", fontsize=12)
plt.text(embeddings[33, 0], embeddings[33, 1], "Officer", fontsize=12)

plt.tight_layout()
plt.savefig("assets/pytorch_gcn_karate.png")
print("Saved assets/pytorch_gcn_karate.png")
