import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuration
os.makedirs('assets', exist_ok=True)
torch.manual_seed(42)

# 1. Defining Custom Self-Attention Class (for education)
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        
        # Output layer (optional in simple demo, but standard in Transformer)
        self.W_o = nn.Linear(d_head, d_model, bias=False)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        
        # Projections
        Q = self.W_q(x) # [batch, seq, d_head]
        K = self.W_k(x) # [batch, seq, d_head]
        V = self.W_v(x) # [batch, seq, d_head]
        
        # Scaled Dot-Product Attention
        # scores: [batch, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Output
        context = torch.matmul(attn_weights, V) # [batch, seq, d_head]
        
        # Output projection (back to d_model)
        output = self.W_o(context)
        
        return output, attn_weights

# 2. Input Data
# Batch size = 1, Seq len = 4, Embedding dim = 8
x = torch.randn(1, 4, 8)
words = ["The", "cat", "sat", "down"]

# 3. Running the Model
d_model = 8
d_head = 8 # For simplicity d_head = d_model (Single Head)
model = SimpleSelfAttention(d_model, d_head)

output, attn_weights = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention Weights shape: {attn_weights.shape}")

# 4. Visualization
# Extract weights for the first element in the batch
weights = attn_weights[0].detach().numpy()

plt.figure(figsize=(6, 5))
plt.imshow(weights, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Attention Weight')
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.title("Self-Attention Heatmap (PyTorch Module)")
plt.xlabel("Key")
plt.ylabel("Query")

for i in range(len(words)):
    for j in range(len(words)):
        plt.text(j, i, f"{weights[i, j]:.2f}", 
                 ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig('assets/pytorch_attention_heatmap.png')
print("Saved visualization to assets/pytorch_attention_heatmap.png")

# 5. Comparison with nn.MultiheadAttention (built-in)
print("\n--- Comparison with nn.MultiheadAttention ---")
multihead_attn = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)
# Forward in nn.MultiheadAttention returns (attn_output, attn_output_weights)
# Note: attn_output_weights is returned only if need_weights=True (default True)
mh_out, mh_weights = multihead_attn(x, x, x) # Q, K, V are the same (Self-Attention)
print(f"nn.MultiheadAttention Output shape: {mh_out.shape}")
print(f"nn.MultiheadAttention Weights shape: {mh_weights.shape}")
