import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuration
os.makedirs('assets', exist_ok=True)
torch.manual_seed(42)

# 1. Data
# We'll use a slightly richer corpus to see clusters
corpus_text = """
king queen prince princess
man woman boy girl
apple orange banana fruit
car bus train vehicle
"""
# Simple processing
sentences = [s.split() for s in corpus_text.strip().split('\n')]
vocab = set([w for s in sentences for w in s])
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# 2. Preparing Pairs (Skip-Gram)
pairs = []
window_size = 1
for sentence in sentences:
    for i, target in enumerate(sentence):
        for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
            if i == j: continue
            pairs.append((word2idx[target], word2idx[sentence[j]]))

# Convert to tensors
target_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
context_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)

print(f"Number of training pairs: {len(pairs)}")

# 3. Word2Vec Model (Skip-Gram) in PyTorch
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # W1: Embeddings (Input -> Hidden)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # W2: Context (Hidden -> Output)
        # In practice, Negative Sampling is often used, here full Softmax (Linear)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, target_word):
        # target_word: [batch_size]
        embeds = self.embeddings(target_word) # [batch_size, embedding_dim]
        out = self.linear(embeds)             # [batch_size, vocab_size]
        return out

embedding_dim = 2 # Force 2D for easy visualization
model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training
EPOCHS = 2000
print("Starting Word2Vec training (PyTorch)...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(target_tensor)
    loss = criterion(output, context_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Visualization
# Extract embedding weights
trained_embeddings = model.embeddings.weight.data.numpy()

plt.figure(figsize=(10, 10))
for i, word in enumerate(vocab):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.text(x + 0.02, y + 0.02, word, fontsize=12)

plt.title("Word Embeddings (PyTorch nn.Embedding)")
plt.grid(True)
plt.savefig('assets/pytorch_word2vec.png')
print("Saved visualization to assets/pytorch_word2vec.png")

# Let's check "arithmetic" (approximate, since 2D and small corpus is limited, but let's try)
def get_vec(w):
    return trained_embeddings[word2idx[w]]

print("\nSample vectors:")
print(f"King: {get_vec('king')}")
print(f"Queen: {get_vec('queen')}")
