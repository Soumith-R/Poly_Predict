# polymer_property_predictor/models/transformer/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, layer_scale_init=1e-4):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        # Layer scaling
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(d_model))

    def forward(self, x, mask=None, relative_pos=None):
        # Pre-norm architecture
        normalized = self.norm1(x)
        attn_output, _ = self.self_attention(
            normalized, normalized, normalized,
            key_padding_mask=mask,
            need_weights=False
        )
        x = x + self.dropout(self.gamma1 * attn_output)
        
        normalized = self.norm2(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(self.gamma2 * ff_output)
        
        return x

class PolymerEncoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model=256, 
        n_layers=6, 
        n_heads=8, 
        d_ff=1024, 
        dropout=0.1,
        max_seq_length=1000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Sinusoidal positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_pos_encoding(max_seq_length, d_model)
        )
        
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_seq_length - 1))
        
        # Encoder layers with gradient checkpointing
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Initialize parameters
        self._init_parameters()

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe

    def _init_parameters(self):
        # Xavier uniform for embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Initialize relative position bias
        nn.init.normal_(self.rel_pos_bias, std=0.02)
        
        # Initialize layer parameters
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def create_attention_mask(self, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if src_mask is None:
            return None
        # Convert boolean mask to float attention mask
        return src_mask.logical_not()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = src.shape
        device = src.device
        
        # Input embedding
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create attention mask
        attention_mask = self.create_attention_mask(src_mask)
        
        # Process through encoder layers with gradient checkpointing
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask,
                    self.rel_pos_bias
                )
            else:
                x = layer(x, attention_mask, self.rel_pos_bias)
        
        return self.norm(x)

    def reset_parameters(self):
        """Reset all parameters for initialization or fine-tuning"""
        self._init_parameters()
