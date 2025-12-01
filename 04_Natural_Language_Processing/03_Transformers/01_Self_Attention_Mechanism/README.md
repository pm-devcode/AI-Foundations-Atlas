# Self-Attention Mechanism

## 1. Idea
Previous architectures (RNN, LSTM) processed sequences step-by-step, which hindered parallelism and capturing long-range dependencies. **Transformer** (Vaswani et al., 2017) introduced the **Self-Attention** mechanism, which allows the model to "look" at all words in the sequence simultaneously and decide which are important for the currently processed word.

## 2. Query, Key, Value
Each word (embedding) is projected into three vectors using learnable weight matrices ($W^Q, W^K, W^V$):
*   **Query (Q)**: What am I looking for? (representation of the current word)
*   **Key (K)**: What do I offer? (label/index of the word we are comparing against)
*   **Value (V)**: What is my content? (information we will pass on if Query matches Key)

## 3. Scaled Dot-Product Attention
Attention formula:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

1.  **$QK^T$**: We calculate similarity (dot product) between the query (Query) and all keys (Keys). The result is a matrix of "raw" attention scores.
2.  **$/ \sqrt{d_k}$**: Scaling (dividing by the square root of the key dimension) to prevent gradients from saturating in Softmax for large values.
3.  **Softmax**: Normalizing results to a probability distribution (sum = 1). These are our **Attention Weights**.
4.  **$\cdot V$**: Weighted sum of values (Values). Words we pay attention to have a large influence on the result.

## 4. Implementation
*   **00_scratch.py**: Manual implementation of the mechanism in NumPy. Step-by-step walkthrough of matrix multiplication and Softmax.
*   **01_pytorch.py**: Implementation of the `SelfAttention` class in PyTorch and usage of the built-in `nn.MultiheadAttention`.
