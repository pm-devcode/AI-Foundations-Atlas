import numpy as np

def train_bigram_model(text):
    """
    Trains a simple Bigram model (Markov Chain).
    """
    # 1. Create Vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Vocabulary: {''.join(chars)}")

    # 2. Count Transitions
    # N[i, j] = count of character j following character i
    N = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for i in range(len(text) - 1):
        ch1 = text[i]
        ch2 = text[i+1]
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
        
    # 3. Normalize to get Probabilities
    # P[i, j] = Probability of j given i
    # Add pseudocount for smoothing (avoid zero probability)
    P = (N + 1).astype(np.float32)
    P /= P.sum(axis=1, keepdims=True)
    
    return P, stoi, itos

def generate_text(P, stoi, itos, start_char=' ', num_chars=100):
    """
    Generates text using the trained Bigram model.
    """
    ix = stoi.get(start_char, 0)
    out = [start_char]
    
    for _ in range(num_chars):
        # Get probability distribution for the next character
        p = P[ix]
        
        # Sample from the distribution
        ix = np.random.choice(len(p), p=p)
        out.append(itos[ix])
        
    return "".join(out)

if __name__ == "__main__":
    # Tiny Shakespeare-like dataset
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """
    
    print("--- Training Bigram Model (Scratch) ---")
    P, stoi, itos = train_bigram_model(text)
    
    print("\n--- Generating Text ---")
    print(f"Input context: (None - Bigram only looks at the last character)")
    generated = generate_text(P, stoi, itos, start_char='T', num_chars=200)
    print(f"Generated:\n{generated}")
    
    print("\n--- Analysis ---")
    print("Notice that the text is nonsensical but locally coherent (e.g., vowels often follow consonants).")
    print("This is because the model has no 'memory' beyond the previous character.")
