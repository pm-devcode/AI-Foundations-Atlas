import re
import collections

class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, text):
        # 1. Pre-tokenization (simple whitespace split + char level)
        # Add </w> to mark end of word
        words = text.split()
        vocab = collections.defaultdict(int)
        for word in words:
            vocab[' '.join(list(word)) + ' </w>'] += 1
            
        self.vocab = vocab
        
        num_merges = self.vocab_size - len(vocab) # Simplified logic for target size
        # Actually, usually we run for N merges. Let's run for fixed N merges for demo.
        num_merges = 20 
        
        print(f"Starting BPE Training (Target merges: {num_merges})...")
        
        for i in range(num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            self.merges[best] = ''.join(best)
            
            print(f"Merge #{i+1}: {best} -> {''.join(best)}")

    def encode(self, text):
        # Simplified inference
        # In real BPE, we apply merges in order of priority
        # Here we just show the final vocab state for the training text
        pass

# --- Testing ---
corpus = """
low low low low low lower lower newest newest newest newest newest newest wide wide wide
"""

tokenizer = BPETokenizer()
tokenizer.train(corpus)

print("\nFinal Vocabulary State:")
for word, freq in tokenizer.vocab.items():
    print(f"{word}: {freq}")
