# Generative AI

## Overview
Generative AI focuses on creating new data instances that resemble the training data. Unlike discriminative models (which classify data), generative models learn the underlying probability distribution of the data to generate novel samples.

## Key Categories

### 1. Autoencoders
Unsupervised learning of efficient data codings.
*   **Vanilla Autoencoder**: Compressing data into a latent space and reconstructing it.
*   **VAE (Variational Autoencoder)**: Introducing probabilistic latent spaces to enable smooth generation of new data.

### 2. GANs (Generative Adversarial Networks)
A game-theoretic approach where two networks compete.
*   **DCGAN (Deep Convolutional GAN)**: Using CNNs in the Generator and Discriminator to generate realistic images.

### 3. Diffusion Models
The current state-of-the-art for image generation.
*   **DDPM (Denoising Diffusion Probabilistic Models)**: Learning to reverse a gradual noise addition process to generate data from pure noise.
