# Convolutional Neural Networks (CNN)

## 1. Executive Summary
Convolutional Neural Networks (CNNs) are the foundation of modern Computer Vision. Unlike Multi-Layer Perceptrons (MLPs) that flatten images into a single vector (losing spatial information), CNNs process images as 2D or 3D grids. They use learnable filters to automatically detect features like edges, textures, and shapes.

## 2. Historical Context
CNNs were inspired by biological research on the visual cortex by **Hubel and Wiesel** (1959). **Yann LeCun** pioneered the practical application of CNNs with **LeNet-5** in **1998**, used for reading zip codes on mail. However, the true revolution arrived in **2012** with **AlexNet**, which dominated the ImageNet competition, proving that deep CNNs could outperform traditional computer vision techniques by a large margin.

## 3. Real-World Analogy
Think of **scanning a document with a magnifying glass**.
*   **Convolution**: You don't read the whole page at once. You slide a small window (the magnifying glass/kernel) over the text, focusing on one small area at a time.
*   **Feature Detection**: In that small window, you recognize simple patterns (lines, curves).
*   **Hierarchy**: As you process more, your brain combines these lines into letters, letters into words, and words into sentences. CNNs work similarly: early layers see edges, deeper layers see objects (eyes, wheels, faces).

## 4. Key Concepts

1.  **Convolution**: Sliding a kernel (filter) over the image to compute a dot product.
    $$ (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n) $$
2.  **Pooling**: Downsampling the image to reduce computation and make the model invariant to small shifts (e.g., Max Pooling takes the largest value in a window).
3.  **Stride**: How many pixels the filter moves at each step.
4.  **Padding**: Adding zeros around the border to preserve image dimensions.

## 5. Implementation Details

1.  **`00_scratch.py`**: Demonstration of the "Forward Pass" operations.
    *   We create a synthetic image.
    *   Manually implement `convolve2d` and `max_pool`.
    *   Apply edge detection filters (Sobel).
2.  **`01_pytorch.py`**: Full training of a simple CNN.
    *   Dataset: Synthetic images of vertical and horizontal lines.
    *   The network learns to classify these patterns.
    *   We visualize the learned filters.

## 6. Results

### Scratch Convolution (Edge Detection)
![Scratch CNN Operations](assets/scratch_cnn_ops.png)

### PyTorch Learned Filters
![PyTorch CNN Filters](assets/pytorch_cnn_filters.png)

## 7. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
