# polymer_property_predictor/models/tokenizer/polymer_tokenizer.py

import torch
from typing import List, Dict, Optional
from collections import defaultdict
import re
from transformers import BertTokenizer  # Changed from PreTrainedTokenizerFast

class ChemicalTokenizer:
    """
    A chemical-aware tokenizer for polymer SMILES strings that preserves chemical meaning
    and identifies important substructures.
    """
    def __init__(self, max_length=150):
        self.max_length = max_length
        self.special_tokens = {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'CLS': '[CLS]',
            'SEP': '[SEP]'
        }
        self.vocab = self._create_base_vocab()
        self.pad_token_id = self.vocab[self.special_tokens['PAD']]
        self.cls_token_id = self.vocab[self.special_tokens['CLS']]
        self.sep_token_id = self.vocab[self.special_tokens['SEP']]
        self.unk_token_id = self.vocab[self.special_tokens['UNK']]

    def _create_base_vocab(self) -> Dict[str, int]:
        """Create basic vocabulary with special tokens and common chemical symbols"""
        vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        
        # Add common chemical elements and symbols
        chemical_symbols = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I',
                          '=', '#', '-', '(', ')', '[', ']', '*', '+', '@']
        
        for symbol in chemical_symbols:
            if symbol not in vocab:
                vocab[symbol] = len(vocab)
        
        return vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """Encode text to token ids"""
        tokens = self._tokenize(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
            
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad sequences
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        elif padding_length < 0:
            token_ids = token_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            
        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Convert SMILES string into tokens"""
        tokens = []
        i = 0
        while i < len(text):
            # Handle two-character tokens first
            if i + 1 < len(text) and text[i:i+2] in self.vocab:
                tokens.append(text[i:i+2])
                i += 2
            # Handle single-character tokens
            elif text[i] in self.vocab:
                tokens.append(text[i])
                i += 1
            # Handle unknown characters
            else:
                tokens.append(self.special_tokens['UNK'])
                i += 1
        return tokens

    def decode(self, 
               token_ids: torch.Tensor,
               skip_special_tokens: bool = True) -> str:
        """
        Decode token indices back to SMILES string.
        
        Args:
            token_ids: Tensor of token indices
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded SMILES string
        """
        tokens = []
        for idx in token_ids:
            token = self.inverse_vocab[idx.item()]
            if skip_special_tokens and token in self.special_tokens.values():
                continue
            tokens.append(token)
        
        # Post-process to ensure valid SMILES
        smiles = ''.join(tokens)
        return smiles

    def save_vocabulary(self, path: str):
        """Save the tokenizer vocabulary to a file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    @classmethod
    def from_vocabulary(cls, path: str, max_length: int = 150):
        """Load a tokenizer from a saved vocabulary file."""
        import json
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab=vocab, max_length=max_length)
