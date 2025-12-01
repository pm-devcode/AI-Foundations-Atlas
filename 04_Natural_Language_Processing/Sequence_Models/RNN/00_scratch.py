import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Vanilla RNN Cell ---
class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Weights initialization (Xavier/Glorot)
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01 # Input to Hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to Hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 # Hidden to Output
        
        self.bh = np.zeros((hidden_size, 1)) # Hidden bias
        self.by = np.zeros((output_size, 1)) # Output bias

    def forward(self, inputs, h_prev):
        """
        inputs: list of input vectors (one-hot) for the sequence
        h_prev: initial hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        loss = 0
        
        # Forward pass through time
        for t, x in enumerate(inputs):
            xs[t] = np.zeros((self.Wxh.shape[1], 1))
            xs[t][x] = 1 # One-hot encoding
            
            # Update hidden state
            # h_t = tanh(Wxh * x_t + Whh * h_{t-1} + b_h)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            
            # Compute output (unnormalized log probabilities)
            # y_t = Why * h_t + b_y
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            
            # Softmax probability
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
        return xs, hs, ps

# --- Data: "hello" sequence ---
# We want to learn: h -> e, e -> l, l -> l, l -> o
data = "hello"
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

print(f"Data: {data}")
print(f"Vocab: {chars}")

# Inputs and Targets (Indices)
inputs = [char_to_idx[ch] for ch in data[:-1]] # h, e, l, l
targets = [char_to_idx[ch] for ch in data[1:]] # e, l, l, o

# --- Run Forward Pass (Untrained) ---
rnn = RNNCell(vocab_size, hidden_size=10, output_size=vocab_size)
h0 = np.zeros((10, 1))
xs, hs, ps = rnn.forward(inputs, h0)

# --- Visualization of Hidden States ---
# We visualize how the hidden state changes over time steps
hidden_states = np.concatenate([hs[t] for t in range(len(inputs))], axis=1)

plt.figure(figsize=(8, 6))
plt.imshow(hidden_states, cmap='hot', interpolation='nearest')
plt.title("RNN Hidden State Activations (Untrained)")
plt.xlabel("Time Step (Input Char)")
plt.ylabel("Hidden Neuron Index")
plt.xticks(range(len(inputs)), list(data[:-1]))
plt.colorbar(label="Activation (tanh)")
plt.savefig("assets/scratch_rnn_hidden.png")
print("Saved assets/scratch_rnn_hidden.png")

# Note: We are not implementing BPTT (Backpropagation Through Time) here fully 
# as it is quite complex for a single file without cluttering. 
# The focus is on the Forward Pass architecture.
