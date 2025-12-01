import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Graph Generation ---
# Let's create a simple graph with 4 nodes
# 0 -- 1
# |    |
# 2 -- 3
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])

# Node Features (H^0)
# Let's say each node has 2 features
X = np.array([
    [1, 0], # Node 0
    [0, 1], # Node 1
    [0, 1], # Node 2
    [1, 0]  # Node 3
])

print("Adjacency Matrix (A):\n", A)
print("Node Features (X):\n", X)

# --- 2. GCN Layer Implementation ---
# Rule: H' = sigma( D^-0.5 * (A + I) * D^-0.5 * H * W )

def gcn_layer(A, H, W):
    # A_hat = A + I (Self-loops)
    I = np.eye(A.shape[0])
    A_hat = A + I
    
    # D_hat (Degree Matrix of A_hat)
    D_hat = np.sum(A_hat, axis=1)
    D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(D_hat))
    
    # Normalized Adjacency: D^-0.5 * A_hat * D^-0.5
    # This normalizes the sum of neighbors so features don't explode
    A_norm = np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
    
    # Aggregation: A_norm * H
    # Each node gets a weighted sum of its neighbors' features
    agg = np.dot(A_norm, H)
    
    # Transformation: agg * W
    z = np.dot(agg, W)
    
    # Activation: ReLU
    return np.maximum(0, z)

# --- 3. Forward Pass ---
# Initialize Weights
np.random.seed(42)
input_dim = X.shape[1]
hidden_dim = 2
output_dim = 2

W1 = np.random.randn(input_dim, hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim)

print("\nWeights W1:\n", W1)

# Layer 1
H1 = gcn_layer(A, X, W1)
print("\nHidden Features (Layer 1):\n", H1)

# Layer 2
H2 = gcn_layer(A, H1, W2)
print("\nOutput Features (Layer 2):\n", H2)

# --- 4. Visualization ---
G = nx.from_numpy_array(A)
pos = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

plt.figure(figsize=(10, 4))

# Plot Graph Structure
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_weight='bold')
plt.title("Graph Structure")

# Plot Feature Space (Output)
plt.subplot(1, 2, 2)
plt.scatter(H2[:, 0], H2[:, 1], c=['red', 'blue', 'blue', 'red'], s=200)
for i in range(4):
    plt.text(H2[i, 0], H2[i, 1], str(i), fontsize=12, ha='center', va='center', color='white')
plt.title("Output Feature Space (GCN Embedding)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)

plt.tight_layout()
plt.savefig("assets/scratch_gcn.png")
print("Saved assets/scratch_gcn.png")

# Explanation:
# Nodes 0 and 3 are structurally similar (connected to 1 and 2).
# Nodes 1 and 2 are structurally similar (connected to 0 and 3).
# The GCN should map similar nodes to similar points in the embedding space.
