# Vanilla Autoencoder

## 1. Executive Summary
An Autoencoder is an unsupervised neural network trained to reconstruct its input. It consists of an **Encoder** that compresses the input into a lower-dimensional **Latent Space**, and a **Decoder** that attempts to reconstruct the original input from this compressed representation. It forces the network to learn the most efficient features of the data.

## 2. Historical Context
Autoencoders have been studied since the 1980s (e.g., by **Geoffrey Hinton** and the PDP group). They gained massive popularity in **2006** when Hinton and Salakhutdinov showed that "Deep Belief Networks" (stacked autoencoders) could be pre-trained layer by layer, effectively solving the vanishing gradient problem and kickstarting the **Deep Learning revolution**.

## 3. Real-World Analogy
Think of **MP3 Compression** or **Zip Files**.
*   **Input**: A large raw audio file (WAV).
*   **Encoder**: The MP3 algorithm removes frequencies humans can't hear and compresses the data into a small file.
*   **Latent Code**: The `.mp3` file itself. It's much smaller than the original but contains the "essence" of the song.
*   **Decoder**: The music player reads the MP3 and plays back the sound.
*   **Reconstruction**: The sound isn't *exactly* the same as the original studio recording (it's "lossy"), but it's close enough that you recognize the song perfectly.

## 4. Key Concepts

1.  **Encoder**: Compresses input $x$ to latent vector $z$.
    $$ z = f(x) $$
2.  **Latent Space (Bottleneck)**: The compressed representation. Its dimension is much smaller than the input.
3.  **Decoder**: Reconstructs $\hat{x}$ from $z$.
    $$ \hat{x} = g(z) $$
4.  **Reconstruction Loss**: Measures how close the output is to the input (e.g., MSE).

## 5. Implementation Details

1.  **`00_scratch.py`**: Simulation of a "linear autoencoder" using SVD (Singular Value Decomposition) in NumPy. Demonstrates that a linear autoencoder is equivalent to PCA (Principal Component Analysis).
2.  **`01_pytorch.py`**: Implementation of a non-linear Autoencoder (with ReLU) in PyTorch. Trained on the Digits dataset (8x8 images).

## 6. Results

### Latent Space Visualization
![Latent Space](assets/pytorch_latent_space.png)

*The 2D representation of the 64-dimensional digits. Note how similar digits (same colors) cluster together.*

### Reconstruction Quality
![Reconstruction](assets/pytorch_reconstruction.png)

*Top: Original images. Bottom: Reconstructed images after compression.*

## 7. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
