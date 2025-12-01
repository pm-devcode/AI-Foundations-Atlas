# ResNet (Residual Networks)

## 1. Concept
ResNet introduced the concept of **Residual Learning** to solve the vanishing gradient problem in very deep neural networks.

### The Problem
As networks get deeper, accuracy gets saturated and then degrades rapidly. This is not caused by overfitting but by optimization difficulties (gradients vanish or explode).

### The Solution: Skip Connections
Instead of learning the underlying mapping $H(x)$ directly, ResNet learns the residual function $F(x) = H(x) - x$.
The original mapping is recast into $F(x) + x$.

*   **Hypothesis**: It is easier to optimize the residual mapping than to optimize the original, unreferenced mapping.
*   **Identity Mapping**: If the optimal function is the identity, the weights can easily be driven to zero.

## 2. Architecture
### Bottleneck Block
Used in ResNet-50/101/152 to reduce parameters.
Structure:
1.  $1 \times 1$ Conv (Reduce dimensions)
2.  $3 \times 3$ Conv (Process)
3.  $1 \times 1$ Conv (Restore dimensions - Expansion $\times 4$)

### Configurations
| Model | Layers |
| :--- | :--- |
| ResNet-50 | [3, 4, 6, 3] |
| ResNet-101 | [3, 4, 23, 3] |
| ResNet-152 | [3, 8, 36, 3] |

## 3. Implementation Details
*   **`00_model.py`**: PyTorch implementation of `ResNet` class and `Bottleneck` block. Includes factory functions for ResNet-50, 101, and 152.

## 4. How to Run
```bash
# Verify model architecture
python 00_model.py
```
