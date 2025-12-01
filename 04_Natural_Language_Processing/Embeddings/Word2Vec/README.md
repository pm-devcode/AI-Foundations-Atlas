# Word Embeddings: Word2Vec

## 1. Executive Summary
Word Embeddings are a technique to represent words as dense vectors of real numbers. Unlike sparse representations (like One-Hot Encoding), embeddings capture semantic relationships: words with similar meanings are close to each other in the vector space.

## 2. Historical Context
**Word2Vec** was introduced by **Tomas Mikolov** and his team at Google in **2013**. It was a breakthrough because it allowed for the efficient training of high-quality embeddings on massive datasets. It popularized the idea that "King - Man + Woman = Queen", showing that simple algebraic operations on these vectors could reveal semantic analogies.

## 3. Real-World Analogy
Think of **Map Coordinates**.
*   **One-Hot**: Saying "Paris is City #45" and "Berlin is City #92". It tells you nothing about where they are relative to each other.
*   **Embeddings**: Giving the GPS coordinates (Latitude, Longitude).
    *   Paris: [48.85, 2.35]
    *   Berlin: [52.52, 13.40]
    *   Now you can calculate the distance. You can see that Paris is closer to Berlin than to Tokyo.
    *   Word2Vec learns these "coordinates" for words based on how they are used in sentences.

## 4. Key Concepts

1.  **CBOW (Continuous Bag of Words)**: Predicts the middle word based on the context (surrounding words).
2.  **Skip-Gram**: Predicts context words based on the middle word. (Used in our example).
3.  **Embedding Space**: A high-dimensional space where semantic similarity = geometric proximity.

## 5. Mathematics (Skip-Gram)

For a given input word $w_I$ (one-hot), the network has one hidden layer (no activation) and an output layer with Softmax.

1.  **Input -> Hidden**: Selecting a row from the weight matrix $W_{in}$ (this is our embedding!).
    $$ h = W_{in}^T \cdot x $$
2.  **Hidden -> Output**: Multiplication by the context matrix $W_{out}$.
    $$ u = W_{out}^T \cdot h $$
3.  **Softmax**: Probability of each word in the vocabulary appearing as context.
    $$ P(w_O | w_I) = \text{softmax}(u) $$

## 6. Implementation Details

1.  **`00_scratch.py`**: Manual implementation of Skip-Gram in NumPy. We build a vocabulary, create training pairs, and update weights $W_{in}$ (embeddings) and $W_{out}$ using Gradient Descent.
2.  **`01_pytorch.py`**: Using `nn.Embedding` in PyTorch to learn vectors on a simple corpus and visualize similarity.

## 7. Results

### PyTorch Word Embeddings
![Word Embeddings](assets/pytorch_word2vec.png)

*Note: In the visualization, words like "king", "queen", "prince" should cluster together, separate from "apple", "orange".*

## 8. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
