import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Data Generation ---
# Non-linear time series: Sine wave + Trend + Noise
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 50, n_samples)
y = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, n_samples)

# Plot Data
plt.figure(figsize=(12, 5))
plt.plot(y)
plt.title("Synthetic Time Series (Sine + Trend + Noise)")
plt.savefig("assets/lstm_data.png")
print("Saved assets/lstm_data.png")

# --- 2. Preprocessing ---
# LSTMs are sensitive to scale, so we normalize to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
y_norm = scaler.fit_transform(y.reshape(-1, 1))

# Create Sequences
# Input: [y_{t-seq_len}, ..., y_{t-1}]
# Output: y_t
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 50
X, y_target = create_sequences(y_norm, SEQ_LENGTH)

# Train/Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_target[:train_size], y_target[train_size:]

# Convert to Tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 3. LSTM Model ---
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = TimeSeriesLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 4. Training ---
EPOCHS = 100
losses = []

print("Training LSTM...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.5f}")

# --- 5. Evaluation & Forecasting ---
model.eval()
with torch.no_grad():
    train_preds = model(X_train).numpy()
    test_preds = model(X_test).numpy()

# Inverse Transform to get original scale
train_preds = scaler.inverse_transform(train_preds)
y_train_orig = scaler.inverse_transform(y_train.numpy())
test_preds = scaler.inverse_transform(test_preds)
y_test_orig = scaler.inverse_transform(y_test.numpy())

# Plot Results
plt.figure(figsize=(14, 6))

# Plot Training Data
plt.plot(np.arange(len(y_train_orig)), y_train_orig, label='True Train')
plt.plot(np.arange(len(train_preds)), train_preds, label='Pred Train', linestyle='--')

# Plot Test Data (shifted)
test_offset = len(y_train_orig)
plt.plot(np.arange(test_offset, test_offset + len(y_test_orig)), y_test_orig, label='True Test')
plt.plot(np.arange(test_offset, test_offset + len(test_preds)), test_preds, label='Pred Test', linestyle='--')

plt.title("LSTM Time Series Forecasting")
plt.legend()
plt.savefig("assets/lstm_forecast.png")
print("Saved assets/lstm_forecast.png")
