# Multi-Layer Perceptron (MLP) - Multiclass Classification

## 1. Executive Summary
Multiclass classification is the task of classifying instances into one of three or more classes. While binary classification answers "Yes/No", multiclass classification answers "Which one?". This requires specific changes to the network architecture, specifically the output layer and loss function.

## 2. Historical Context
The generalization of binary classification to multiple classes using the **Softmax** function (a generalization of the logistic function) became a standard component of neural network training in the 1990s. This enabled the application of neural networks to tasks like handwritten digit recognition (MNIST, 10 classes) and later ImageNet (1000 classes).

## 3. Real-World Analogy
Think of a **Mail Sorting Center**.
*   **Binary Classification**: A worker deciding if a letter is "Spam" or "Not Spam".
*   **Multiclass Classification**: A worker sorting letters into specific bins: "Bills", "Personal Letters", "Advertisements", "Magazines".
*   The worker (Network) looks at the envelope (Input), evaluates the features, and assigns a probability to each bin. The letter goes into the bin with the highest probability.

## 4. Key Concepts

1.  **Output Layer**: Number of neurons equals the number of classes $C$.
2.  **Softmax Activation**: Converts raw scores (logits) into a probability distribution.
    $$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}} $$
    The sum of all outputs is 1.
3.  **One-Hot Encoding**: Representing labels as vectors (e.g., Class 1 is `[0, 1, 0]`).

## 5. Mathematics (Categorical Cross-Entropy)

The loss function for multiclass problems. It penalizes the model if it assigns low probability to the correct class.

$$ L = - \sum_{i=1}^{C} y_i \log(\hat{y}_i) $$

Where $y$ is the one-hot encoded true label and $\hat{y}$ is the predicted probability.

## 6. Implementation Details

1.  **`00_scratch.py`**: Implementation of MLP with matrix support (batch processing), Softmax function, and Cross-Entropy derivative.
2.  **`01_pytorch.py`**: Implementation in PyTorch using `nn.CrossEntropyLoss` (which combines `LogSoftmax` and `NLLLoss` for numerical stability).

## 7. Results

### Scratch Implementation (Multiclass Boundaries)
![Scratch MLP Multiclass](assets/scratch_mlp_multiclass.png)

### PyTorch Implementation (Multiclass Boundaries)
![PyTorch MLP Multiclass](assets/pytorch_mlp_multiclass.png)

## 8. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
