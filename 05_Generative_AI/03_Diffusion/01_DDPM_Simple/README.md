# Denoising Diffusion Probabilistic Models (DDPM)

## 1. Introduction
Diffusion models are a class of generative models that learn to generate data by reversing a gradual noise addition process. Unlike GANs, which learn an implicit distribution, or VAEs, which learn an approximate posterior, Diffusion models learn the gradients of the data distribution (score matching) or, equivalently, how to denoise an image step-by-step.

## 2. Historical Context
*   **The Origins:** The concept was introduced by **Sohl-Dickstein et al.** in 2015 ("Deep Unsupervised Learning using Nonequilibrium Thermodynamics").
*   **The Renaissance:** It remained relatively obscure until **Ho et al.** (2020) demonstrated high-quality image synthesis with "Denoising Diffusion Probabilistic Models" (DDPM), showing they could beat GANs. This led to the explosion of models like DALL-E 2 and Stable Diffusion.

## 3. Real-World Analogy
### Rewinding the Ink Drop
Imagine dropping a single drop of blue ink into a glass of clear water.
*   **Forward Process (Diffusion):** Over time, the ink spreads out until the water is uniformly pale blue. The structure of the drop is lost to entropy. This is easy to simulate.
*   **Reverse Process (Generation):** Now, imagine watching a video of this process in reverse. You see the pale blue water spontaneously gathering molecules together to form a sharp, defined drop. This is what the model learns: how to take "chaos" (noise) and carefully push it back into "order" (an image).

## 4. Core Concepts

### 4.1 The Forward Process (Diffusion)
The forward process is a fixed Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1, \dots, \beta_T$.

Given a data point $x_0 \sim q(x_0)$, the forward process at step $t$ is:
$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}) $$

A key property is that we can sample $x_t$ at any timestep $t$ directly from $x_0$:
$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}) $$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.

### 2.2 The Reverse Process (Denoising)
The reverse process is a learned Markov chain where a neural network predicts the parameters of the reverse transition:
$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

In the simplified DDPM objective (Ho et al., 2020), we fix the variance and train a network $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ that was added to $x_0$ to get $x_t$.

### 2.3 Training Objective
The loss function is simply the Mean Squared Error (MSE) between the actual noise added and the predicted noise:
$$ L_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} [ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 ] $$

## 3. Architecture (U-Net)

The standard architecture for the noise predictor $\epsilon_\theta$ is a **U-Net**.
- **Input**: Noisy image $x_t$ and time embedding $t$.
- **Output**: Predicted noise $\epsilon$ (same shape as input).
- **Time Embedding**: Sinusoidal embeddings (like in Transformers) are injected into the network to tell it which timestep $t$ it is currently denoising.

## 4. Sampling Algorithm

1. Sample $x_T \sim \mathcal{N}(0, \mathbf{I})$.
2. For $t = T, \dots, 1$:
   - Sample $z \sim \mathcal{N}(0, \mathbf{I})$ (if $t > 1$, else $z=0$).
   - Compute $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)) + \sigma_t z$.
3. Return $x_0$.
