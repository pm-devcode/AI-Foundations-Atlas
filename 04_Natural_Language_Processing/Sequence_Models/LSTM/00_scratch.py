import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Concatenated size for weights (input + hidden)
        concat_size = input_size + hidden_size
        
        # Weights for Forget Gate (f)
        self.Wf = np.random.randn(hidden_size, concat_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        
        # Weights for Input Gate (i)
        self.Wi = np.random.randn(hidden_size, concat_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        
        # Weights for Candidate Cell State (C_tilde)
        self.Wc = np.random.randn(hidden_size, concat_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        
        # Weights for Output Gate (o)
        self.Wo = np.random.randn(hidden_size, concat_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        """
        x: input vector (input_size, 1)
        h_prev: previous hidden state (hidden_size, 1)
        c_prev: previous cell state (hidden_size, 1)
        """
        # Concatenate input and previous hidden state
        # Shape: (input_size + hidden_size, 1)
        concat_input = np.vstack((h_prev, x))
        
        # 1. Forget Gate
        # "What information are we going to throw away from the cell state?"
        ft = sigmoid(np.dot(self.Wf, concat_input) + self.bf)
        
        # 2. Input Gate
        # "What new information are we going to store in the cell state?"
        it = sigmoid(np.dot(self.Wi, concat_input) + self.bi)
        
        # 3. Candidate Cell State
        # "What are the new candidate values?"
        c_tilde = tanh(np.dot(self.Wc, concat_input) + self.bc)
        
        # 4. Update Cell State
        # Old state * forget factor + New info * input factor
        ct = ft * c_prev + it * c_tilde
        
        # 5. Output Gate
        # "What are we going to output?"
        ot = sigmoid(np.dot(self.Wo, concat_input) + self.bo)
        
        # 6. Update Hidden State
        ht = ot * tanh(ct)
        
        return ht, ct, (ft, it, ot, c_tilde)

# --- Simulation ---
input_size = 4
hidden_size = 8
lstm = LSTMCell(input_size, hidden_size)

# Dummy input sequence (e.g., 5 time steps)
seq_length = 20
inputs = [np.random.randn(input_size, 1) for _ in range(seq_length)]

# Initial states
h_t = np.zeros((hidden_size, 1))
c_t = np.zeros((hidden_size, 1))

# Store gate activations for visualization
forget_gates = []
input_gates = []
output_gates = []
cell_states = []

print("Running LSTM Forward Pass...")
for t, x in enumerate(inputs):
    h_t, c_t, gates = lstm.forward(x, h_t, c_t)
    ft, it, ot, c_tilde = gates
    
    forget_gates.append(ft.flatten())
    input_gates.append(it.flatten())
    output_gates.append(ot.flatten())
    cell_states.append(c_t.flatten())

# --- Visualization ---
forget_gates = np.array(forget_gates).T
input_gates = np.array(input_gates).T
output_gates = np.array(output_gates).T
cell_states = np.array(cell_states).T

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot Forget Gate
im0 = axes[0].imshow(forget_gates, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[0].set_title("Forget Gate Activations (0=Forget, 1=Keep)")
axes[0].set_ylabel("Neuron")
plt.colorbar(im0, ax=axes[0])

# Plot Input Gate
im1 = axes[1].imshow(input_gates, cmap='Greens', aspect='auto', vmin=0, vmax=1)
axes[1].set_title("Input Gate Activations (0=Ignore, 1=Update)")
axes[1].set_ylabel("Neuron")
plt.colorbar(im1, ax=axes[1])

# Plot Output Gate
im2 = axes[2].imshow(output_gates, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
axes[2].set_title("Output Gate Activations (0=Block, 1=Output)")
axes[2].set_ylabel("Neuron")
plt.colorbar(im2, ax=axes[2])

# Plot Cell State
im3 = axes[3].imshow(cell_states, cmap='RdBu', aspect='auto')
axes[3].set_title("Cell State Values (Memory)")
axes[3].set_ylabel("Neuron")
axes[3].set_xlabel("Time Step")
plt.colorbar(im3, ax=axes[3])

plt.tight_layout()
plt.savefig("assets/scratch_lstm_gates.png")
print("Saved assets/scratch_lstm_gates.png")
