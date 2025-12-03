# AI-Foundations-Atlas

> **"Don't just import it. Implement it."**

## Project Mission
**AI-Foundations-Atlas** is an interactive textbook designed to demystify Artificial Intelligence by peeling back the layers of abstraction. 

In an era of "Black Box" APIs and high-level frameworks, this project takes a step back to understand the **"Why"** and **"How"**. We prioritize understanding the mathematical intuition and historical context over finding the fastest production-ready implementation.

## Repository Structure
The project is organized by domain, progressing from foundational statistics to modern Deep Learning architectures.

### 01. Classical Machine Learning
Foundational algorithms rooted in statistics and optimization.
*   **Regression**: Linear Regression, Logistic Regression.
*   **Classification**: KNN, SVM, Naive Bayes.
*   **Clustering**: K-Means, DBSCAN, Gaussian Mixture Models (GMM).
*   **Anomaly Detection**: Isolation Forest.
*   **Dimensionality Reduction**: PCA, LDA.
*   **Ensemble Methods**: Decision Trees, Random Forests, Gradient Boosting.

### 02. Neural Networks Basics
The building blocks of Deep Learning.
*   **The Neuron**: Perceptron (Logic Gates).
*   **Feed Forward**: Multilayer Perceptron (MLP) for binary and multiclass tasks.
*   **Optimization**: SGD, Momentum, Adam.

### 03. Computer Vision
Processing visual data from pixels to objects.
*   **Convolutional Networks**: Basic CNN architecture.
*   **Object Detection**: YOLO v1 (You Only Look Once).
*   **Segmentation**: U-Net.
*   **Architectures**: ResNet (Residual Networks).

### 04. Natural Language Processing
Teaching machines to understand and generate text.
*   **Embeddings**: Word2Vec.
*   **Tokenization**: Byte Pair Encoding (BPE).
*   **Sequence Models**: RNN, LSTM, Seq2Seq with Attention.
*   **Transformers**: Self-Attention Mechanism (The "Transformer" core), Simple GPT (Mini-LLM).

### 05. Generative AI
Creating new data distributions.
*   **Autoencoders**: Vanilla AE, Variational Autoencoder (VAE).
*   **GANs**: Deep Convolutional GAN (DCGAN).
*   **Diffusion**: Denoising Diffusion Probabilistic Models (DDPM).

### 06. Reinforcement Learning
Learning through trial and reward.
*   **Value Based**: Q-Learning (Table), Deep Q-Network (DQN).
*   **Policy Based**: REINFORCE (Policy Gradient).

### 08. Graph Neural Networks
Learning on non-Euclidean data structures.
*   **Graph Convolution**: GCN (Spectral).
*   **Graph Attention**: GAT.

### 09. Time Series Analysis
Analyzing temporal data.
*   **Statistical**: ARIMA.
*   **Deep Learning**: LSTM for forecasting.

## How to Use This Atlas

Each algorithm module is self-contained and typically includes:
1.  **`README.md`**: The "Textbook" chapter explaining the math, history, and intuition.
2.  **`00_scratch.py`**: The "White Box" implementation using **only** `numpy`. No frameworks allowed. This reveals the internal mechanics.
3.  **`01_pytorch.py` / `01_sklearn.py`**: The "Modern" implementation showing how to achieve the same result using industry-standard libraries.

### Example Workflow
1.  Navigate to a module, e.g., `01_Classical_Machine_Learning/Regression/Linear_Regression_Simple`.
2.  Read the `README.md` to understand the Gradient Descent derivation.
3.  Run the scratch implementation to see the math in action:
    ```bash
    python 00_scratch.py
    ```
4.  Compare it with the framework implementation:
    ```bash
    python 01_sklearn.py
    ```

## Hardware Requirements & GPU Recommendation

Most **Classical Machine Learning** algorithms (Section 01) are lightweight and run instantly on any standard CPU.

However, for the **Deep Learning** sections (02-09), specifically:
*   Computer Vision (CNNs, YOLO, ResNet)
*   NLP (Transformers, LSTMs)
*   Generative AI (GANs, Diffusion)

**It is highly recommended to run these on a machine with a capable GPU.** 
While the code will run on a CPU, training times for models like ResNet or DCGAN may be significantly longer. The PyTorch scripts are configured to automatically detect and use a GPU if available (supporting CUDA for NVIDIA, ROCm for AMD, or MPS for macOS).

## Prerequisites

*   Python 3.8+
*   NumPy, Pandas, Matplotlib
*   PyTorch (for Deep Learning modules)
*   Scikit-Learn (for Classical ML comparisons)

## License
This project is for **educational purposes**. The code is designed for readability and understanding, not for production efficiency.
