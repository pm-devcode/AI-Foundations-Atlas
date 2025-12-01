# Variational Autoencoder (VAE)

## 1. Concept
A Variational Autoencoder (VAE) is a generative model that learns a probabilistic mapping between the input space and a latent space.

### Autoencoder vs VAE
*   **Autoencoder**: Maps input to a fixed vector. Latent space is discontinuous. Good for compression, bad for generation.
*   **VAE**: Maps input to a **distribution** (mean $\mu$ and variance $\sigma^2$). Latent space is continuous. Good for generation.

## 2. Mathematical Foundation

### The Objective
Maximize the Evidence Lower Bound (ELBO):
$$ \mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z)) $$

1.  **Reconstruction Loss** ($\mathbb{E}[\log p(x|z)]$): Measures how well the decoder recovers the input.
2.  **KL Divergence** ($D_{KL}$): Regularizes the latent space to be close to a standard normal distribution $\mathcal{N}(0, I)$.

### Reparameterization Trick
To backpropagate through the random sampling $z \sim \mathcal{N}(\mu, \sigma^2)$, we express $z$ as:
$$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
Now the randomness is in $\epsilon$ (which is fixed during backprop), and gradients can flow through $\mu$ and $\sigma$.

## 3. Implementation Details
*   **`00_model.py`**: PyTorch implementation of a Linear VAE.
    *   `encode`: Returns $\mu$ and $\log(\sigma^2)$.
    *   `reparameterize`: Implements the trick.
    *   `decode`: Reconstructs image from $z$.
    *   `loss_function`: BCE + KLD.

## 4. How to Run
```bash
python 00_model.py
```
