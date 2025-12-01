import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

def softmax(x):
    e_x = np.exp(x - np.max(x)) # Stability trick
    return e_x / e_x.sum(axis=0)

class AttentionMechanism:
    def __init__(self):
        pass

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (hidden_size, 1) - The Query
        encoder_outputs: (seq_len, hidden_size) - The Keys/Values
        """
        seq_len, hidden_size = encoder_outputs.shape
        
        # 1. Calculate Alignment Scores
        # Score = Decoder_Hidden . Encoder_Output (Dot Product)
        # Shape: (seq_len, 1)
        scores = np.dot(encoder_outputs, decoder_hidden)
        
        # 2. Calculate Attention Weights (Softmax)
        # Shape: (seq_len, 1)
        attn_weights = softmax(scores)
        
        # 3. Calculate Context Vector
        # Context = Sum(Weights * Encoder_Outputs)
        # Shape: (hidden_size, 1)
        # (hidden_size, seq_len) x (seq_len, 1) -> (hidden_size, 1)
        context_vector = np.dot(encoder_outputs.T, attn_weights)
        
        return context_vector, attn_weights

# --- Simulation ---
hidden_size = 64
seq_len = 10

# Simulate Encoder Outputs (Keys/Values)
# Imagine a sentence with 10 words, each represented by a 64-dim vector
encoder_outputs = np.random.randn(seq_len, hidden_size)

# Simulate Decoder Hidden State (Query)
# The current state of the translator
decoder_hidden = np.random.randn(hidden_size, 1)

# Initialize Attention
attention = AttentionMechanism()

# Run Attention
context, weights = attention.forward(decoder_hidden, encoder_outputs)

print(f"Encoder Outputs Shape: {encoder_outputs.shape}")
print(f"Decoder Hidden Shape: {decoder_hidden.shape}")
print(f"Context Vector Shape: {context.shape}")
print(f"Attention Weights Shape: {weights.shape}")

# --- Visualization ---
plt.figure(figsize=(10, 4))
plt.bar(range(seq_len), weights.flatten(), color='skyblue', edgecolor='black')
plt.title("Attention Weights (Where is the model looking?)")
plt.xlabel("Encoder Time Step (Input Word Index)")
plt.ylabel("Attention Score")
plt.xticks(range(seq_len))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the max attention
max_idx = np.argmax(weights)
plt.bar(max_idx, weights[max_idx], color='orange', edgecolor='black', label='Focus')
plt.legend()

plt.savefig("assets/scratch_attention_weights.png")
print("Saved assets/scratch_attention_weights.png")

# --- Explanation ---
# If weight[3] is high, it means the Decoder is paying a lot of attention 
# to the 4th word in the input sequence to generate the next output word.
