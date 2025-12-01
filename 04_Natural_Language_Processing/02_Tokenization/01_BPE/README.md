# Byte Pair Encoding (BPE)

## 1. Concept
Byte Pair Encoding (BPE) is a subword tokenization algorithm commonly used in NLP models like GPT, RoBERTa, and BART.

### The Problem
*   **Word-level tokenization**: Huge vocabulary, many "Unknown" (UNK) tokens for rare words.
*   **Character-level tokenization**: Sequences become too long, model loses semantic meaning of words.

### The Solution: Subwords
BPE iteratively merges the most frequent pair of adjacent characters (or subwords) into a new subword.
*   Common words remain as whole words.
*   Rare words are broken down into meaningful subwords (e.g., "playing" -> "play" + "ing").

## 2. Algorithm
1.  **Initialize**: Split all words into characters. Add a special end-of-word symbol `</w>`.
2.  **Count**: Count frequency of all adjacent pairs of symbols.
3.  **Merge**: Find the most frequent pair (A, B) and merge it into a new symbol AB.
4.  **Repeat**: Repeat steps 2-3 for a fixed number of merges (hyperparameter).

## 3. Implementation Details
*   **`00_scratch.py`**: Implements the BPE training loop.
    *   `get_stats`: Counts pair frequencies.
    *   `merge_vocab`: Updates the vocabulary by merging the best pair.

## 4. How to Run
```bash
python 00_scratch.py
```
