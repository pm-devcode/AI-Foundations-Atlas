import torch
import torch.nn as nn
import torch.nn.functional as F
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

labels = []
for i in range(num_nodes):
    club = G.nodes[i]['club']
    labels.append(0 if club == 'Mr. Hi' else 1)
labels = torch.tensor(labels, dtype=torch.long)

# Adjacency Matrix (Binary)
A = nx.to_numpy_array(G)
A = torch.tensor(A, dtype=torch.float32)
# Add self-loops
A = A + torch.eye(num_nodes)

# Features: Identity
X = torch.eye(num_nodes, dtype=torch.float32)

# --- 2. GAT Layer (PyTorch) ---
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # W: Transformation matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # a: Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (N, in_features)
        # adj: (N, N) Adjacency matrix
        
        Wh = torch.mm(h, self.W) # (N, out_features)
        N = Wh.size()[0]
        
        # Attention Mechanism
        # We need to compute a^T [Wh_i || Wh_j] for all pairs
        # Repeat Wh to form pairs
        # a_input: (N, N, 2*out_features)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        
        # e: (N, N, 1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Masked Attention: Only compute for neighbors (where adj > 0)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregation
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime), attention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
        self.gat2 = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha)
        
    def forward(self, x, adj):
        x, attn1 = self.gat1(x, adj)
        x, attn2 = self.gat2(x, adj)
        return F.log_softmax(x, dim=1), x

# --- 3. Training ---
model = GAT(nfeat=num_nodes, nhid=8, nclass=2)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[0] = True
train_mask[33] = True

losses = []
print("Training GAT on Karate Club...")

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output, _ = model(X, A)
    loss = F.nll_loss(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --- 4. Visualization ---
model.eval()
_, embeddings = model(X, A)
embeddings = embeddings.detach().numpy()

plt.figure(figsize=(12, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("GAT Training Loss")
plt.xlabel("Epoch")

# Plot 2: Embeddings
plt.subplot(1, 2, 2)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='coolwarm', s=100, edgecolors='k')
plt.title("GAT Node Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.text(embeddings[0, 0], embeddings[0, 1], "Mr. Hi", fontsize=12)
plt.text(embeddings[33, 0], embeddings[33, 1], "Officer", fontsize=12)

plt.tight_layout()
plt.savefig("assets/pytorch_gat_karate.png")
print("Saved assets/pytorch_gat_karate.png")
