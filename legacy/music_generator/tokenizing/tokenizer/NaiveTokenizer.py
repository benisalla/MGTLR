from typing import List, Dict, Optional

class NaiveTokenizer:
    def __init__(self, text_data: str, eost: Optional[str] = None):
        """
        Initializes the tokenizer with unique tokens from the given text data and optional end-of-sequence token"""
        self.tokens = sorted(set(text_data))
        self.stoi = {s: i for i, s in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

        if eost:
            self.stoi[eost] = self.vocab_size
            self.vocab_size += 1

        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text_string: str) -> List[int]:
        """
        Encodes a string into a list of integers based on the tokenizer's vocabulary"""
        return [self.stoi[char] for char in text_string if char in self.stoi]

    def decode(self, token_arr: List[int]) -> str:
        """
        Decodes a list of integers back into a string"""
        return ''.join(self.itos[token] for token in token_arr if token in self.itos)