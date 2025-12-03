# Generative AI

## Overview
Generative AI focuses on creating new data instances that resemble the training data. Unlike discriminative models (which classify data), generative models learn the underlying probability distribution of the data to generate novel samples.

## Key Categories

### Autoencoders
Unsupervised learning of efficient data codings.
*   **VAE (Variational Autoencoder)**: Introducing probabilistic latent spaces to enable smooth generation of new data.
*   **Vanilla Autoencoder**: Compressing data into a latent space and reconstructing it.

### Diffusion Models
The current state-of-the-art for image generation.
*   **DDPM (Denoising Diffusion Probabilistic Models)**: Learning to reverse a gradual noise addition process to generate data from pure noise.

### GANs (Generative Adversarial Networks)
A game-theoretic approach where two networks compete.
*   **DCGAN (Deep Convolutional GAN)**: Using CNNs in the Generator and Discriminator to generate realistic images.

## Note on Text Generation
For Generative Large Language Models (LLMs) like GPT, please refer to the **Natural Language Processing** section under **Transformers**.
