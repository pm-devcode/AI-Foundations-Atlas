# Neural Networks Basics

## Overview
This section covers the foundational building blocks of Deep Learning. Before diving into complex architectures like Transformers or GANs, it is crucial to understand the single neuron, how networks learn via backpropagation, and the optimization algorithms that make training possible.

## Key Categories

### Feed Forward Networks (MLP)
*   **Multilayer Perceptron (MLP)**: Stacking neurons in layers allows the network to approximate non-linear functions (Universal Approximation Theorem).
    *   **Binary Classification**: Using Sigmoid activation.
    *   **Multiclass Classification**: Using Softmax activation.

### Optimization
The engine of learning. How do we update weights to minimize error?
*   **Adam**: Adaptive Moment Estimation. The default optimizer for most modern deep learning tasks.
*   **SGD (Stochastic Gradient Descent)**: The fundamental update rule.
*   **Momentum**: Accelerating SGD in the relevant direction and dampening oscillations.

### The Neuron
*   **Perceptron**: The simplest artificial neuron. A linear classifier that mimics the logic gates (AND, OR, NOT). It forms the basis for all subsequent neural networks.
