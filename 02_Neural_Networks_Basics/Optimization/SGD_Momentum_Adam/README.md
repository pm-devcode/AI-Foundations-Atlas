# Optimization Algorithms: SGD, Momentum, Adam

## 1. Executive Summary
Optimization algorithms are the engines of neural network training. Their goal is to find the set of parameters (weights) that minimize the Loss Function. While **Gradient Descent** is the foundation, modern deep learning relies on more advanced variants like **Momentum** and **Adam** to navigate complex loss landscapes efficiently and avoid getting stuck in local minima.

## 2. Historical Context
*   **Stochastic Gradient Descent (SGD)**: Roots trace back to **Robbins and Monro (1951)**. It became the standard for training neural networks in the 1980s (Backpropagation).
*   **Momentum**: Introduced by **Boris Polyak** in **1964** to accelerate convergence in areas where the gradient is small or oscillates.
*   **Adam (Adaptive Moment Estimation)**: Proposed by **Diederik Kingma and Jimmy Ba** in **2014**. It combines the benefits of Momentum and RMSProp (adaptive learning rates) and is currently the default optimizer for most deep learning tasks.

## 3. Real-World Analogy
Imagine you are trying to get to the bottom of a dark valley (Loss Function).
*   **SGD**: You take a step, look at the slope, and take another step. If the ground is bumpy, you jitter around a lot (like a **drunk person walking**).
*   **Momentum**: You roll a **heavy ball** down the hill. It gains speed. If there's a small bump, the momentum carries it over. It doesn't turn sharply, smoothing out the path.
*   **Adam**: You are in a **smart rover**. It has separate speed controls for each wheel (parameter). If the terrain is steep in one direction (high gradient), it slows down to be careful. If it's flat (low gradient), it speeds up. It also remembers its previous speed (momentum).

## 4. Key Concepts

1.  **Learning Rate ($\eta$)**: The step size. Too small = slow convergence. Too large = divergence.
2.  **Gradient ($\nabla J$)**: The direction of steepest ascent. We move in the opposite direction ($-\nabla J$).
3.  **Momentum**: Accumulating a velocity vector to smooth out updates.
4.  **Adaptive Learning Rates**: Adjusting the step size for each parameter individually based on historical gradient statistics (Adam).

## 5. Mathematics

### 1. Stochastic Gradient Descent (SGD)
$$ w_{t+1} = w_t - \eta \nabla J(w_t) $$

### 2. SGD with Momentum
We introduce velocity $v$:
$$ v_{t+1} = \gamma v_t + \eta \nabla J(w_t) $$
$$ w_{t+1} = w_t - v_{t+1} $$
Where $\gamma$ is the momentum coefficient (usually 0.9).

### 3. Adam
Adam maintains moving averages of the gradients ($m$) and squared gradients ($v$).

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$

Bias correction:
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

Update:
$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

## 6. Implementation Details

1.  **`00_scratch.py`**: Manual implementation of the update rules. We minimize the function $f(x, y) = x^2 + 10y^2$ (an elongated bowl).
2.  **`01_pytorch.py`**: Using `torch.optim.SGD` and `torch.optim.Adam`.

## 7. Results

### Scratch Implementation
![Scratch Optimization](assets/scratch_optimization.png)

*   **SGD (Red)**: Oscillates heavily along the steep y-axis and moves slowly along the flat x-axis.
*   **Momentum (Blue)**: Dampens oscillations and accelerates towards the minimum.
*   **Adam (Green)**: Takes a direct path, adjusting for the different scales of x and y.

### PyTorch Implementation
![PyTorch Optimization](assets/pytorch_optimization.png)

## 8. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
