import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Configuration ---
input_size = 1
hidden_size = 32
num_layers = 1
output_size = 1
learning_rate = 0.01
n_epochs = 200
seq_length = 20 # Length of input sequence

# --- Data Generation (Sine Wave) ---
def generate_sine_wave(seq_len, num_samples):
    x = np.linspace(0, 100, num_samples)
    y = np.sin(x)
    return y

data = generate_sine_wave(seq_length, 1000)

# Create sequences
# Input: [x_t, x_{t+1}, ..., x_{t+seq_len-1}]
# Target: x_{t+seq_len}
X, y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = np.array(X).reshape(-1, seq_length, 1) # (Samples, Seq_Len, Input_Size)
y = np.array(y).reshape(-1, 1)             # (Samples, Output_Size)

# Convert to Tensors
X_train = torch.from_numpy(X[:800]).float()
y_train = torch.from_numpy(y[:800]).float()
X_test = torch.from_numpy(X[800:]).float()
y_test = torch.from_numpy(y[800:]).float()

# --- Model Definition ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM Layer
        # batch_first=True -> (Batch, Seq, Feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (Batch, Seq, Hidden)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = LSTMPredictor(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
losses = []
print("Training LSTM on Sine Wave...")
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.6f}')

# --- Evaluation & Visualization ---
model.eval()
with torch.no_grad():
    test_preds = model(X_test).numpy()

plt.figure(figsize=(12, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot Predictions
plt.subplot(1, 2, 2)
plt.plot(y_test.numpy(), label='True Value')
plt.plot(test_preds, label='LSTM Prediction', linestyle='--')
plt.title("Sine Wave Prediction (Test Set)")
plt.legend()

plt.tight_layout()
plt.savefig("assets/pytorch_lstm_sine.png")
print("Saved assets/pytorch_lstm_sine.png")
