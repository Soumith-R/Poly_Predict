import torch
from tokenizer_code import ChemicalTokenizer

class TokenizerWrapper:
    def __init__(self, max_length=150):
        self.tokenizer = ChemicalTokenizer(max_length=max_length)
        self.max_length = max_length
        
    def __call__(self, text):
        # Use the tokenizer's encode method directly
        return self.tokenizer.encode(text, add_special_tokens=True)
