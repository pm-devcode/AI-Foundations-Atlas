# The Perceptron

## 1. Executive Summary
The Perceptron is the fundamental building block of deep learning. It is the simplest model of a biological neuron, acting as a linear binary classifier. While limited in its capabilities (it can only solve linearly separable problems), understanding the Perceptron is essential for understanding how modern massive neural networks function.

## 2. Historical Context
The Perceptron was invented by **Frank Rosenblatt** in **1957** at the Cornell Aeronautical Laboratory. He built the **Mark I Perceptron**, a physical machine designed for image recognition. It created a wave of optimism for AI. However, in **1969**, Marvin Minsky and Seymour Papert published the book *"Perceptrons"*, mathematically proving that a single-layer perceptron could not solve the **XOR problem**. This revelation cooled funding and interest in neural networks, leading to the first "AI Winter".

## 3. Real-World Analogy
Think of a **Thermostat** or a simple voting system.
*   **Inputs**: Several sensors (Temperature, Humidity, User Presence).
*   **Weights**: How important is each sensor? (Temperature is very important, Humidity less so).
*   **Bias**: The threshold setting (e.g., "Turn on if sum > 20").
*   **Activation**: If the weighted sum of inputs exceeds the threshold, the heater turns **ON** (1); otherwise, it stays **OFF** (0).

## 4. Key Concepts

1.  **Inputs ($x$)**: Feature vector.
2.  **Weights ($w$)**: Importance of each input.
3.  **Bias ($b$)**: Shift allowing activation even with zero inputs.
4.  **Weighted Sum ($z$)**:
    $$ z = w \cdot x + b = \sum_{i=1}^{n} w_i x_i + b $$
5.  **Activation Function ($\phi$)**: Heaviside step function.
    $$ \hat{y} = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{otherwise} \end{cases} $$

## 5. The XOR Problem
The Perceptron is a **linear classifier**. It draws a straight line to separate classes.
*   **AND / OR gates**: Linearly separable (can be solved).
*   **XOR gate**: Not linearly separable (cannot be solved by a single line).

## 6. Implementation Details

1.  **`00_scratch.py`**: Implementation of the Perceptron from scratch. We train it on AND, OR gates (success) and try XOR (failure).
2.  **`01_pytorch.py`**: Implementation of a single neuron using PyTorch (`nn.Linear` + `Sigmoid`).

## 7. Results

### Scratch Implementation (Logic Gates Boundaries)
![Perceptron Logic Gates](assets/perceptron_gates.png)

*Note: For XOR (rightmost), the perceptron fails to find a separating line.*

## 8. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
