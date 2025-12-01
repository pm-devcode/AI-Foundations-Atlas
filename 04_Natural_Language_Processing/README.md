# Natural Language Processing (NLP)

## Overview
Natural Language Processing gives computers the ability to understand, interpret, and generate human language. This section traces the evolution from static word embeddings to dynamic, context-aware Transformer models.

## Key Categories

### 1. Embeddings
Representing words as dense vectors where similar words have similar geometry.
*   **Word2Vec**: Learning vector representations of words from large corpora using Skip-gram or CBOW architectures.

### 2. Tokenization
Breaking text into smaller units for processing.
*   **BPE (Byte Pair Encoding)**: A subword tokenization method that balances vocabulary size and out-of-vocabulary handling. Standard in modern LLMs.

### 3. Sequence Models
Handling data where order matters.
*   **RNN (Recurrent Neural Networks)**: Networks with loops to persist information.
*   **LSTM (Long Short-Term Memory)**: Solving the vanishing gradient problem in RNNs to capture long-term dependencies.
*   **Seq2Seq with Attention**: Encoder-Decoder architectures for translation, introducing the concept of "Attention" to focus on relevant parts of the input.

### 4. Transformers
The modern standard for NLP.
*   **Self-Attention Mechanism**: The core component of the Transformer, allowing the model to weigh the importance of different words in a sentence regardless of their distance.
