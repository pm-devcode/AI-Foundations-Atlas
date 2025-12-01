import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Configuration ---
SOS_token = 0
EOS_token = 1
hidden_size = 128
vocab_size = 20 # Small vocab for demo
max_length = 10
learning_rate = 0.01
n_epochs = 500 # Short training for demo

# --- Data Generation ---
# Task: Reverse the sequence
# Input: [2, 5, 9] -> Output: [9, 5, 2]
def generate_pair():
    seq_len = random.randint(3, 8)
    seq = [random.randint(2, vocab_size-1) for _ in range(seq_len)]
    input_tensor = torch.tensor(seq, dtype=torch.long).view(-1, 1)
    target_tensor = torch.tensor(seq[::-1] + [EOS_token], dtype=torch.long).view(-1, 1)
    return input_tensor, target_tensor

# --- Model Definition ---

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculate Attention Weights
        # attn_weights: (1, max_length)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        # Apply Attention to Encoder Outputs
        # attn_applied: (1, 1, hidden_size)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # Combine Embedded Input and Context Vector
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

# --- Training ---
encoder = EncoderRNN(vocab_size, hidden_size)
decoder = AttnDecoderRNN(hidden_size, vocab_size)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

loss_history = []

print("Training Seq2Seq with Attention...")
for epoch in range(n_epochs):
    input_tensor, target_tensor = generate_pair()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    # Encoder Pass
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder Pass
    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden # Use last encoder hidden state

    # Teacher Forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    loss_history.append(loss.item() / target_length)
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item() / target_length:.4f}")

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Seq2Seq Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("assets/pytorch_seq2seq_loss.png")
print("Saved assets/pytorch_seq2seq_loss.png")

# --- Evaluation (Show Attention) ---
def evaluate_and_show_attention(input_seq):
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.long).view(-1, 1)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
            
        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(str(topi.item()))
            
            decoder_input = topi.detach()
            
        return decoded_words, decoder_attentions[:di+1]

# Test
test_seq = [3, 7, 12, 4]
print(f"\nTest Input: {test_seq}")
output_words, attentions = evaluate_and_show_attention(test_seq)
print(f"Output: {output_words}")

# Plot Attention Matrix
plt.figure()
plt.matshow(attentions.numpy())
plt.title("Attention Matrix")
plt.xlabel("Input Sequence")
plt.ylabel("Output Sequence")
plt.savefig("assets/pytorch_attention_matrix.png")
print("Saved assets/pytorch_attention_matrix.png")
