import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Configuration
os.makedirs('assets', exist_ok=True)
np.random.seed(42)

# 1. Data (Small corpus)
text = "king queen man woman prince princess apple orange"
corpus = [text.split()]

# 2. Vocabulary Construction
words = sorted(list(set(text.split())))
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for i, w in enumerate(words)}
vocab_size = len(words)
print(f"Vocabulary: {words}")

# 3. Generating Training Pairs (Skip-Gram)
# Window size = 1 (word to the left and right)
pairs = []
for sentence in corpus:
    for i, target_word in enumerate(sentence):
        target_idx = word2idx[target_word]
        # Context: window +/- 1
        for j in range(max(0, i - 1), min(len(sentence), i + 2)):
            if i == j: continue
            context_word = sentence[j]
            context_idx = word2idx[context_word]
            pairs.append((target_idx, context_idx))

print(f"Training pairs (Target -> Context): {[(idx2word[t], idx2word[c]) for t, c in pairs[:5]]}...")

# 4. Weight Initialization
embedding_dim = 2 # Set to 2D for easy plotting
LR = 0.1
EPOCHS = 1000

# W1: Input -> Hidden (These are our embeddings!)
# Dimension: [Vocab_Size, Embedding_Dim]
W1 = np.random.randn(vocab_size, embedding_dim) * 0.01

# W2: Hidden -> Output (Context)
# Dimension: [Embedding_Dim, Vocab_Size]
W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

# 5. Training (Stochastic Gradient Descent - sample by sample)
losses = []

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print("Starting Word2Vec training (NumPy)...")
for epoch in range(EPOCHS):
    loss_sum = 0
    
    # Shuffle pairs
    np.random.shuffle(pairs)
    
    for target_idx, context_idx in pairs:
        # --- Forward Pass ---
        # 1. Hidden Layer: Select row from W1 (Lookup)
        # x is one-hot, so W1.T * x is simply the row W1[target_idx]
        h = W1[target_idx] # Shape: (embedding_dim,)
        
        # 2. Output Layer
        u = np.dot(W2.T, h) # Shape: (vocab_size,)
        
        # 3. Softmax
        y_pred = softmax(u) # Shape: (vocab_size,)
        
        # --- Loss (Cross Entropy) ---
        # L = -log(y_pred[context_idx])
        loss_sum += -np.log(y_pred[context_idx] + 1e-9)
        
        # --- Backward Pass ---
        # Gradient w.r.t u: y_pred - y_true
        # y_true is one-hot vector with 1 at context_idx
        e = y_pred.copy()
        e[context_idx] -= 1 # Error vector
        
        # Gradient for W2: dL/dW2 = h * e^T
        # h: (dim,), e: (vocab,) -> (dim, vocab)
        dW2 = np.outer(h, e)
        
        # Gradient for W1 (only for target_idx row): dL/dh * dh/dW1
        # dL/dh = W2 * e
        # W2: (dim, vocab), e: (vocab,) -> (dim,)
        dh = np.dot(W2, e)
        
        # Update weights
        W2 -= LR * dW2
        W1[target_idx] -= LR * dh # Update only the embedding of the input word
        
    if epoch % 100 == 0:
        avg_loss = loss_sum / len(pairs)
        losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# 6. Visualization
plt.figure(figsize=(8, 8))
for i, word in enumerate(words):
    x, y = W1[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=12)

plt.title("Word Embeddings (NumPy Scratch Implementation)")
plt.grid(True)
plt.savefig('assets/numpy_word2vec.png')
print("Saved visualization to assets/numpy_word2vec.png")
