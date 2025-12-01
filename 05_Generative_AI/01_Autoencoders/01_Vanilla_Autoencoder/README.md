# Vanilla Autoencoder

## 1. Idea
An Autoencoder is a neural network trained in an unsupervised manner, aiming to learn an efficient representation of data (encoding) by attempting to reconstruct the input at the output.

The architecture consists of two parts:
1.  **Encoder**: Compresses the input $x$ into a hidden vector $z$ (latent vector) of smaller dimension.
    $$ z = f(x) $$
2.  **Decoder**: Reconstructs an approximation of the input $\hat{x}$ based on the code $z$.
    $$ \hat{x} = g(z) $$

## 2. Latent Space
The vector $z$ resides in the so-called Latent Space. Since the dimension of $z$ is smaller than $x$ (bottleneck), the network is forced to learn the most important features of the data, ignoring noise.

## 3. Loss Function
The goal is to minimize the reconstruction error, typically MSE (Mean Squared Error):
$$ L = ||x - \hat{x}||^2 $$

## 4. Implementation
*   **00_scratch.py**: Simulation of a "linear autoencoder" using SVD (Singular Value Decomposition) in NumPy. Demonstrates the mathematical foundation of dimensionality reduction (equivalent to PCA).
*   **01_pytorch.py**: Implementation of a classic Autoencoder with linear layers and ReLU activation in PyTorch. Trained on the Digits dataset (8x8 images).
