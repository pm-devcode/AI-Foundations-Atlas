# Self-Attention Mechanism

## 1. Executive Summary
Self-Attention is the core mechanism of the Transformer architecture. It allows a model to process a sequence of data (like words in a sentence) and decide, for each word, which other words in the sequence are relevant. This enables the model to capture long-range dependencies and context much better than previous sequential models like RNNs.

## 2. Historical Context
The Self-Attention mechanism was introduced by **Ashish Vaswani et al.** (Google Brain) in the landmark paper **"Attention Is All You Need"** in **2017**. This paper proposed the **Transformer** architecture, which dispensed with recurrence and convolutions entirely, relying solely on attention mechanisms. This shift led to the development of massive language models like BERT, GPT-3, and ChatGPT.

## 3. Real-World Analogy
Think of the **Cocktail Party Effect**.
*   **Scenario**: You are at a noisy party (the input sequence).
*   **Query**: You want to listen to a specific friend (what you are focusing on).
*   **Keys**: You scan the room, looking at everyone's faces (matching your focus against available sources).
*   **Attention Score**: When you see your friend, the match is high. For a stranger, the match is low.
*   **Value**: You focus your hearing on your friend's voice (the information you extract), tuning out the background noise.
*   Self-Attention does this for *every word simultaneously*, allowing each word to "listen" to every other word to understand the full context.

## 4. Key Concepts

1.  **Query (Q)**: What am I looking for? (Representation of the current word).
2.  **Key (K)**: What do I offer? (Label/Index of the word we are comparing against).
3.  **Value (V)**: What is my content? (Information to pass on if Query matches Key).
4.  **Scaled Dot-Product Attention**:
    $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

## 5. Implementation Details

1.  **`00_scratch.py`**: Manual implementation of the mechanism in NumPy. Step-by-step walkthrough of matrix multiplication and Softmax.
2.  **`01_pytorch.py`**: Implementation of the `SelfAttention` class in PyTorch and usage of the built-in `nn.MultiheadAttention`.

## 6. Results

### PyTorch Attention Heatmap
![Attention Heatmap](assets/pytorch_attention_heatmap.png)

*The heatmap shows how much attention each word pays to every other word. Darker colors indicate higher attention.*

## 7. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
