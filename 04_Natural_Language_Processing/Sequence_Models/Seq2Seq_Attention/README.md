# Sequence-to-Sequence with Attention

## 1. Overview
Sequence-to-Sequence (Seq2Seq) models are designed to map a fixed-length input sequence to a fixed-length output sequence, where the lengths can differ (e.g., translating "I love you" [3 words] to "Je t'aime" [2 words]).

The **Attention Mechanism** was introduced to solve the bottleneck problem of the basic Encoder-Decoder architecture. Instead of forcing the Encoder to compress the entire source sentence into a single fixed-size context vector, Attention allows the Decoder to "look back" at specific parts of the source sentence at each step of the generation process.

### Key Components
*   **Encoder:** Processes the input sequence and produces a sequence of hidden states (keys/values).
*   **Decoder:** Generates the output sequence one token at a time.
*   **Attention:** Calculates a weighted sum of the Encoder's hidden states (Context Vector) based on the Decoder's current state (Query).

## 2. Historical Context
*   **Seq2Seq (2014):** Sutskever et al. introduced the Encoder-Decoder architecture using LSTMs for machine translation. It relied on a single final hidden state to carry all information.
*   **The Bottleneck:** For long sentences, the performance degraded because the fixed-size vector couldn't hold all the nuances.
*   **Attention (2015):** Bahdanau et al. introduced "Neural Machine Translation by Jointly Learning to Align and Translate". They proposed that the model should learn to align the source and target words dynamically. This is often called "Bahdanau Attention" (Additive).
*   **Luong Attention (2015):** Luong et al. proposed "Effective Approaches to Attention-based Neural Machine Translation", introducing "Dot-Product Attention" (Multiplicative), which is simpler and faster.

## 3. Real-World Analogy
**The Interpreter:**
*   **Standard Seq2Seq:** An interpreter listens to a whole speech, memorizes it perfectly in their head (single vector), and then translates it. If the speech is long, they might forget the beginning.
*   **Seq2Seq with Attention:** An interpreter takes detailed notes (Encoder outputs) while listening. When translating a specific part, they look back at their notes (Attention) to find the exact words the speaker used, ensuring accuracy even for long speeches.

## 4. Mathematical Foundation (Dot-Product Attention)
1.  **Alignment Score:** How relevant is Encoder state $h_s$ to the current Decoder state $h_t$?
    $$ \text{score}(h_t, h_s) = h_t^T \cdot h_s $$
2.  **Attention Weights:** Normalize scores to probabilities.
    $$ \alpha_{ts} = \text{softmax}(\text{score}(h_t, h_s)) $$
3.  **Context Vector:** Weighted sum of Encoder states.
    $$ c_t = \sum_s \alpha_{ts} h_s $$
4.  **Final Output:** Combine context vector with decoder state to predict the next word.

## 5. Implementation Details
*   **`00_scratch.py`**: A NumPy simulation of the Attention Mechanism. It visualizes how the "Context Vector" is formed by taking a weighted sum of "Encoder Outputs" based on a "Decoder Query".
*   **`01_pytorch.py`**: A PyTorch implementation of a Seq2Seq model with Attention for a sequence reversal task. It includes:
    *   `EncoderRNN`: A GRU-based encoder.
    *   `AttnDecoderRNN`: A GRU-based decoder with an attention layer.
    *   **Teacher Forcing:** Using the actual target output as the next input during training to speed up convergence.
    *   **Attention Matrix:** Visualization of the alignment between input and output sequences.

## 6. Impact
Attention is the foundation of the **Transformer** architecture (2017), which replaced Recurrent Networks entirely with "Self-Attention", leading to models like BERT and GPT.

## 7. Results

### Attention Weights (Scratch)
![Attention Weights](assets/scratch_attention_weights.png)
*Visualization of how much attention the decoder pays to each encoder time step.*

### Training Loss (PyTorch)
![Training Loss](assets/pytorch_seq2seq_loss.png)
*Loss curve for the sequence reversal task.*

### Attention Matrix (PyTorch)
![Attention Matrix](assets/pytorch_attention_matrix.png)
*Heatmap showing the alignment between input and output sequences. The diagonal pattern indicates that the model has learned to reverse the sequence correctly (aligning the first output with the last input, etc.).*

