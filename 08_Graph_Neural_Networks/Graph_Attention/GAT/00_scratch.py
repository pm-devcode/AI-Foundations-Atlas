import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha * x, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# --- 1. Setup ---
# Node 0 is the target. Nodes 1, 2, 3 are neighbors.
# Features (Dimension = 3)
h = np.array([
    [1.0, 0.0, 0.0], # Node 0 (Target)
    [0.0, 1.0, 0.0], # Node 1
    [0.0, 0.0, 1.0], # Node 2
    [1.0, 1.0, 1.0]  # Node 3
])

# Weight Matrix W (Transformation)
np.random.seed(42)
W = np.random.randn(3, 2) # Map 3 dim -> 2 dim

# Attention Vector 'a' (2 * output_dim)
a_vec = np.random.randn(4, 1) # 2+2 = 4

print("Node Features h:\n", h)

# --- 2. Linear Transformation ---
# Wh = h * W
Wh = np.dot(h, W)
print("\nTransformed Features Wh:\n", Wh)

# --- 3. Attention Mechanism (For Node 0) ---
# We calculate attention scores for neighbors 1, 2, 3 (and self 0)
neighbors = [0, 1, 2, 3]
scores = []

target_wh = Wh[0] # Wh_i

print("\nCalculating Attention Scores for Node 0:")
for j in neighbors:
    neighbor_wh = Wh[j] # Wh_j
    
    # Concatenate [Wh_i || Wh_j]
    concat = np.concatenate([target_wh, neighbor_wh])
    
    # e_ij = LeakyReLU(a^T * concat)
    e_ij = leaky_relu(np.dot(concat, a_vec))
    scores.append(e_ij[0])
    print(f"  Neighbor {j}: Raw Score e_0{j} = {e_ij[0]:.4f}")

# --- 4. Softmax (Normalization) ---
scores = np.array(scores)
alphas = softmax(scores)
print("\nNormalized Attention Coefficients (Alpha):\n", alphas)

# --- 5. Aggregation ---
# h'_0 = Sum(alpha_0j * Wh_j)
h_prime = np.zeros_like(Wh[0])

for idx, j in enumerate(neighbors):
    h_prime += alphas[idx] * Wh[j]

print("\nNew Feature for Node 0 (h'_0):\n", h_prime)

# --- 6. Visualization ---
plt.figure(figsize=(6, 4))
plt.bar(neighbors, alphas, color=['gray', 'red', 'green', 'blue'])
plt.title("Attention Weights for Node 0")
plt.xlabel("Neighbor ID")
plt.ylabel("Attention Alpha")
plt.xticks(neighbors, ["Self (0)", "Node 1", "Node 2", "Node 3"])
plt.savefig("assets/scratch_gat_attention.png")
print("Saved assets/scratch_gat_attention.png")

# Explanation:
# GAT allows Node 0 to weigh its neighbors differently.
# Unlike GCN (which is fixed average), GAT is data-dependent.
