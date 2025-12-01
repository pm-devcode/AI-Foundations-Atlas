# Multi-Layer Perceptron (MLP) - Binary Classification

## 1. Executive Summary
The Multi-Layer Perceptron (MLP) is a Feed-Forward Neural Network that overcomes the limitations of the simple Perceptron. By adding one or more **hidden layers** with non-linear activation functions, MLPs can learn complex, non-linear decision boundaries, such as solving the XOR problem.

## 2. Historical Context
After the "AI Winter" caused by the limitations of the Perceptron, the field was revitalized in **1986**. **David Rumelhart, Geoffrey Hinton, and Ronald Williams** published the seminal paper on **Backpropagation**. This algorithm provided a mathematically sound way to train multi-layer networks by efficiently calculating gradients, enabling the "Deep Learning" revolution decades later.

## 3. Real-World Analogy
Think of a **Corporate Hierarchy** or an **Assembly Line**.
*   **Input Layer**: Raw data (Junior employees reporting raw facts).
*   **Hidden Layers**: Middle Managers. They don't see the raw data directly but process the reports from juniors, synthesizing them into higher-level insights (features).
*   **Output Layer**: The CEO. Makes the final decision based on the synthesized reports from the managers, not the raw data.
*   **Backpropagation**: If the CEO makes a bad decision (error), the blame flows down the hierarchy, adjusting how each manager and employee operates to avoid the mistake next time.

## 4. Key Concepts

1.  **Hidden Layer**: Layers between input and output. They extract features.
2.  **Non-linearity**: Essential for solving complex problems. Without non-linear activations (ReLU, Sigmoid), a multi-layer network is mathematically equivalent to a single layer.
3.  **Backpropagation**: The algorithm for training. It uses the Chain Rule to propagate error backwards.

## 5. Mathematics (Backpropagation)

We use the **Chain Rule** of calculus to find how the error $L$ changes with respect to each weight $w$.

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w} $$

Where:
*   $L$ - Loss Function (e.g., MSE or Binary Cross Entropy).
*   $\hat{y}$ - Network output (after activation).
*   $z$ - Weighted sum (before activation).
*   $w$ - Weight.

## 6. Implementation Details

1.  **`00_scratch.py`**: Implementation of MLP from scratch with one hidden layer. Includes manual implementation of Forward Pass and Backpropagation (calculating derivatives).
2.  **`01_pytorch.py`**: Implementation using `torch.nn.Sequential`, `nn.Linear`, and `nn.ReLU`/`nn.Sigmoid`.

## 7. Results

### Scratch Implementation (XOR Solution)
![Scratch MLP XOR](assets/scratch_mlp_xor.png)

### PyTorch Implementation (XOR Solution)
![PyTorch MLP XOR](assets/pytorch_mlp_xor.png)

## 8. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
