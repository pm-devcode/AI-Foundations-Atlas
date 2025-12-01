# Word Embeddings: Word2Vec

## 1. Idea
Traditional text representations (e.g., One-Hot Encoding) are sparse and do not preserve semantic relationships (the word "king" is just as distant from "queen" as it is from "toaster").

**Word Embeddings** are dense vectors of fixed length (e.g., 64, 128, 300), where similar words are located close to each other in the vector space.

## 2. Word2Vec (Mikolov et al., 2013)
Word2Vec is a family of neural network architectures for learning such representations. Two main variants:

1.  **CBOW (Continuous Bag of Words)**: Predicts the middle word based on the context (surrounding words).
    *   Input: [The, brown, fox] -> Output: [quick]
2.  **Skip-Gram**: Predicts context words based on the middle word.
    *   Input: [quick] -> Output: [The, brown, fox]

## 3. Mathematics (Skip-Gram)
For a given input word $w_I$ (one-hot), the network has one hidden layer (no activation) and an output layer with Softmax.

1.  **Input -> Hidden**: Selecting a row from the weight matrix $W_{in}$ (this is our embedding!).
    $$ h = W_{in}^T \cdot x $$
2.  **Hidden -> Output**: Multiplication by the context matrix $W_{out}$.
    $$ u = W_{out}^T \cdot h $$
3.  **Softmax**: Probability of each word in the vocabulary appearing as context.
    $$ P(w_O | w_I) = \text{softmax}(u) $$

## 4. Implementation
*   **00_scratch.py**: Manual implementation of Skip-Gram in NumPy. We build a vocabulary, create training pairs, and update weights $W_{in}$ (embeddings) and $W_{out}$ using Gradient Descent.
*   **01_pytorch.py**: Using `nn.Embedding` in PyTorch to learn vectors on a simple corpus and visualize similarity.
