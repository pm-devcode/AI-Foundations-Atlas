# Recurrent Neural Networks (RNN)

## 1. Overview
Recurrent Neural Networks (RNNs) are a class of neural networks designed for processing sequential data. Unlike Feed-Forward networks, RNNs have a "memory" (hidden state) that captures information about what has been calculated so far. This makes them ideal for tasks like language modeling, time series prediction, and speech recognition.

### Key Concepts
*   **Sequential Processing:** Inputs are processed one by one, in order.
*   **Hidden State ($h_t$):** The network's memory, updated at each time step based on the current input and the previous hidden state.
*   **Shared Weights:** The same weight matrices ($W_{xh}, W_{hh}, W_{hy}$) are used at every time step, allowing the network to handle sequences of varying lengths.
*   **Backpropagation Through Time (BPTT):** The training algorithm, which unrolls the network over time and propagates gradients back through the sequence.

## 2. Historical Context
*   **Origins (1980s):** The concept of recurrent connections in neural networks dates back to the 1980s (e.g., Hopfield Networks, 1982).
*   **Elman Networks (1990):** Jeffrey Elman introduced the "Simple Recurrent Network" (SRN), which is the standard "Vanilla RNN" architecture we use today. He demonstrated that these networks could learn temporal structure in language.
*   **The Vanishing Gradient Problem (1990s):** Researchers like Sepp Hochreiter and Yoshua Bengio discovered that training standard RNNs on long sequences was difficult because gradients would either vanish (go to zero) or explode (go to infinity) as they were propagated back through many time steps. This led to the development of LSTMs (1997).

## 3. Real-World Analogy
**Reading a Sentence:**
Imagine reading a sentence word by word. You don't understand the sentence by looking at each word in isolation.
*   When you read the word "bank", your understanding depends on the previous words.
*   If the previous words were "I sat by the river", you know "bank" means a river bank.
*   If the previous words were "I deposited money in the", you know "bank" means a financial institution.
*   Your brain maintains a "context" (hidden state) that is updated with every new word you read, allowing you to make sense of the sequence as a whole.

## 4. Mathematical Foundation
The core update equation for a Vanilla RNN cell is:

$$ h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) $$

$$ y_t = W_{hy} h_t + b_y $$

Where:
*   $x_t$ is the input at time $t$.
*   $h_t$ is the hidden state at time $t$.
*   $h_{t-1}$ is the hidden state from the previous time step.
*   $y_t$ is the output at time $t$.
*   $\tanh$ is the activation function (squashes values between -1 and 1).

## 5. Implementation Details
*   **`00_scratch.py`**: A NumPy implementation of the forward pass of a Vanilla RNN cell. It demonstrates how the hidden state is updated step-by-step and visualizes the internal activations.
*   **`01_pytorch.py`**: A PyTorch implementation using `nn.RNN` to learn a simple character sequence ("hell" -> "ello"). It includes a training loop and loss visualization.

## 6. Limitations
*   **Short-Term Memory:** Vanilla RNNs struggle to retain information over long sequences due to the vanishing gradient problem.
*   **Sequential Computation:** They cannot be parallelized easily (unlike Transformers), making them slower to train on long sequences.

## 7. Results

### Hidden State Visualization (Scratch)
![Hidden State](assets/scratch_rnn_hidden.png)
*Visualization of how the hidden state vector changes as the RNN processes the sequence "hello".*

### Training Loss (PyTorch)
![Training Loss](assets/pytorch_rnn_loss.png)
*Loss curve showing the model learning to predict the next character.*

