import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Configuration ---
input_size = 5  # Vocab size (h,e,l,o) - actually 4 unique chars, but let's be safe
hidden_size = 10
num_layers = 1
learning_rate = 0.01
n_epochs = 100

# --- Data Preparation ---
# Sequence: "hello"
# Input: "hell" -> Output: "ello"
idx_to_char = ['h', 'e', 'l', 'o']
char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
data_input = [0, 1, 2, 2] # h, e, l, l
data_target = [1, 2, 2, 3] # e, l, l, o

# One-hot encoding
input_one_hot = np.zeros((len(data_input), 4)) # Sequence length x Input Size
for t, idx in enumerate(data_input):
    input_one_hot[t][idx] = 1

# Convert to Tensors
# Shape: (Sequence Length, Batch Size, Input Size)
inputs = torch.from_numpy(input_one_hot).float().unsqueeze(1) 
targets = torch.tensor(data_target).long()

# --- Model Definition ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Basic RNN Layer
        # batch_first=False (default) -> (Seq, Batch, Feature)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh')
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        # out shape: (Seq, Batch, Hidden)
        # hidden shape: (Layers, Batch, Hidden)
        out, hidden = self.rnn(x, hidden)
        
        # Reshape out to (Seq * Batch, Hidden) for Linear Layer
        out = out.view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

model = SimpleRNN(input_size=4, hidden_size=hidden_size, output_size=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
losses = []
print("Training RNN...")
for epoch in range(n_epochs):
    optimizer.zero_grad()
    hidden = model.init_hidden()
    
    output, hidden = model(inputs, hidden)
    
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# --- Prediction ---
with torch.no_grad():
    hidden = model.init_hidden()
    output, _ = model(inputs, hidden)
    _, predicted = torch.max(output, 1)
    predicted_chars = [idx_to_char[i] for i in predicted]
    print(f"\nInput: hell")
    print(f"Predicted: {''.join(predicted_chars)}")

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("RNN Training Loss (PyTorch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("assets/pytorch_rnn_loss.png")
print("Saved assets/pytorch_rnn_loss.png")
