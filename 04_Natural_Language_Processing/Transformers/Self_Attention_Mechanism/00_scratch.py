import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuration
os.makedirs('assets', exist_ok=True)
np.random.seed(42)

def softmax(x):
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# 1. Input Data Simulation
# Assume we have a 3-word sentence: "I love AI"
# Each word is an embedding vector of dimension d_model = 4
X = np.array([
    [1.0, 0.0, 1.0, 0.0], # I
    [0.0, 2.0, 0.0, 2.0], # love
    [1.0, 1.0, 1.0, 1.0]  # AI
])

seq_len, d_model = X.shape
d_k = 3 # Key/Query dimension (can be different from d_model)
d_v = 3 # Value dimension

print(f"Input shape: {X.shape}")

# 2. Weight Initialization (Wq, Wk, Wv)
# In practice, these are learned by the network
W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_v)

# 3. Linear Projection (Calculating Q, K, V)
Q = np.dot(X, W_Q) # [seq_len, d_k]
K = np.dot(X, W_K) # [seq_len, d_k]
V = np.dot(X, W_V) # [seq_len, d_v]

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# 4. Calculating Scores (QK^T)
scores = np.dot(Q, K.T) # [seq_len, seq_len]
print("\nRaw Scores (QK^T):")
print(scores)

# 5. Scaling (Scaled Dot-Product)
scores_scaled = scores / np.sqrt(d_k)

# 6. Softmax (Attention Weights)
attention_weights = softmax(scores_scaled)
print("\nAttention Weights (Softmax):")
print(np.round(attention_weights, 4))

# 7. Weighted Sum of Values (Output)
output = np.dot(attention_weights, V)
print("\nOutput (Weighted Sum of V):")
print(output)

# 8. Attention Weights Visualization
plt.figure(figsize=(6, 5))
plt.imshow(attention_weights, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Attention Weight')
plt.xticks(range(seq_len), ["I", "love", "AI"])
plt.yticks(range(seq_len), ["I", "love", "AI"])
plt.title("Self-Attention Heatmap (NumPy)")
plt.xlabel("Key (Source)")
plt.ylabel("Query (Target)")

# Add numerical values to the heatmap
for i in range(seq_len):
    for j in range(seq_len):
        plt.text(j, i, f"{attention_weights[i, j]:.2f}", 
                 ha="center", va="center", color="white" if attention_weights[i, j] < 0.5 else "black")

plt.tight_layout()
plt.savefig('assets/numpy_attention_heatmap.png')
print("Saved visualization to assets/numpy_attention_heatmap.png")
