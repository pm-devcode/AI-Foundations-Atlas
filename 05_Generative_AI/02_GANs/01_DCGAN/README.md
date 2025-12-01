# Deep Convolutional GAN (DCGAN)

## 1. Introduction
**Generative Adversarial Networks (GANs)** consist of two neural networks competing against each other in a zero-sum game: a Generator (G) and a Discriminator (D).

## 2. Historical Context
*   **The Inventors:** **Alec Radford, Luke Metz, and Soumith Chintala** (2015).
*   **The Problem:** Before DCGAN, training GANs with Convolutional Neural Networks was notoriously unstable. Models often collapsed or failed to converge.
*   **The Solution:** DCGAN proposed a set of architectural constraints (like using Batch Normalization and removing fully connected layers) that made training stable and reproducible, launching the era of high-quality AI image generation.

## 3. Real-World Analogy
### The Art Forger and the Detective
*   **The Generator (The Forger):** Tries to paint a fake Picasso. At first, he is terrible and just splashes paint randomly.
*   **The Discriminator (The Detective):** Tries to determine if a painting is a real Picasso or a fake.
*   **The Game:**
    1.  The Forger shows a painting.
    2.  The Detective says "Fake! The brushstrokes are wrong."
    3.  The Forger learns from this feedback and tries again.
    4.  Over time, the Forger becomes so good that the Detective can no longer tell the difference (50% guess rate).

## 4. DCGAN Architecture (Radford et al., 2015)
DCGAN introduced architectural constraints to make GAN training stable for images:
*   Replace pooling layers with **strided convolutions** (Discriminator) and **fractional-strided convolutions** (Generator).
*   Use **Batch Normalization** in both G and D.
*   Remove fully connected hidden layers for deeper architectures.
*   Use **ReLU** activation in Generator for all layers except output (which uses Tanh).
*   Use **LeakyReLU** activation in Discriminator for all layers.

## 3. Loss Function (Minimax Loss)
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)} [\log (1 - D(G(z)))] $$

*   **Discriminator** wants to maximize the probability of assigning 1 to real data and 0 to fake data.
*   **Generator** wants to minimize $\log(1 - D(G(z)))$, which is equivalent to maximizing $\log D(G(z))$ (fooling the discriminator).

## 4. Implementation
*   **00_scratch.py**: A simplified 1D GAN simulation in NumPy. The Generator tries to transform uniform noise into a Gaussian distribution, while the Discriminator tries to separate them.
*   **01_pytorch.py**: Full DCGAN implementation in PyTorch training on the MNIST dataset to generate handwritten digits.
