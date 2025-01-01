# polymer_property_predictor/models/transformer/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torch.nn.utils import parametrizations

MODEL_CONFIG = {
    'd_model': 256,
    'n_layers': 6, 
    'n_heads': 8,
    'd_ff': 1024,
    'dropout': 0.1
}

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for combining encoder outputs with decoder states.
    Specialized for polymer property prediction.
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Use fixed dimensions
        self.max_seq_length = 512
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Update relative position bias size
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, self.max_seq_length, self.max_seq_length))
        
        # Additional projections for chemical features
        self.chemical_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_head]))
        
        # Add gating mechanism
        self.gate = nn.Linear(d_model, d_model)
        
        # Add gradient checkpointing for better memory efficiency
        self.gradient_checkpointing = True
        
        # Add layer scale for better convergence
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 0.1)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chemical_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention mechanism.
        
        Args:
            query: Query tensor from decoder [batch_size, tgt_len, d_model]
            key: Key tensor from encoder [batch_size, src_len, d_model]
            value: Value tensor from encoder [batch_size, src_len, d_model]
            mask: Attention mask
            chemical_features: Optional chemical feature tensor
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Pad or truncate sequences to max_seq_length
        query = F.pad(query, (0, 0, 0, self.max_seq_length - query.size(1)))[:,:self.max_seq_length,:]
        key = F.pad(key, (0, 0, 0, self.max_seq_length - key.size(1)))[:,:self.max_seq_length,:]
        value = F.pad(value, (0, 0, 0, self.max_seq_length - value.size(1)))[:,:self.max_seq_length,:]
        
        # Project and reshape to match expected dimensions
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_head)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_head)  
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_head)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head] 
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply chemical features if provided
        if chemical_features is not None:
            chemical_proj = self.chemical_proj(chemical_features)
            K = K + chemical_proj.unsqueeze(1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, device=Q.device))
        scores = scores * self.layer_scale.view(1, 1, 1, -1)
        
        # Apply masking if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Apply attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.shape[1:])
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class DecoderLayer(nn.Module):
    """
    Single decoder layer with chemical property awareness.
    """
    def __init__(self,
                d_model: int = 256,  # Changed from 256 to match optimal value
                n_heads: int = 8,   # Changed from 8 to match optimal value
                d_ff: int = 1024,    # Changed from 1024 to match optimal value
                dropout: float = 0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = CrossAttention(d_model, n_heads, dropout)
        # Cross attention with encoder outputs
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Add stochastic depth for regularization
        self.drop_path = DropPath(dropout)
        
        # Add layer scale initialization
        self.gamma1 = nn.Parameter(torch.ones(d_model) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(d_model) * 0.1)
        
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                chemical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor
            encoder_output: Output from encoder
            self_attn_mask: Self-attention mask
            cross_attn_mask: Cross-attention mask
            chemical_features: Optional chemical features
            
        Returns:
            Output tensor
        """
        # Self attention
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross attention
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output,
            cross_attn_mask, chemical_features
        )
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.ff_net(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        # Return just the output tensor instead of tuple
        return x

class PropertyPredictor(nn.Module):
    """
    Head for predicting specific polymer properties.
    """
    def __init__(self, d_model: int, n_properties: int):
        super().__init__()
        
        self.property_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_properties)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict polymer properties from encoded representation.
        
        Args:
            x: Encoded representation [batch_size, seq_len, d_model]
            
        Returns:
            Property predictions [batch_size, n_properties]
        """
        # Global average pooling
        x = torch.mean(x, dim=1)
        # Predict properties
        return self.property_net(x)

class PolymerDecoder(nn.Module):
    """
    Complete polymer decoder for property prediction and structure generation.
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Match checkpoint dimensions for prediction heads 
        self.cloud_point_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )
        
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 512), 
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Property prediction heads
        head_structure = lambda: nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(), 
            nn.Linear(512, 1)
        )
        
        self.mw_head = head_structure()
        self.pdi_head = head_structure()
        self.phi_head = head_structure()
        self.pressure_head = head_structure()
        
        # Uncertainty heads
        self.mw_uncertainty = head_structure()
        self.pdi_uncertainty = head_structure() 
        self.phi_uncertainty = head_structure()
        self.pressure_uncertainty = head_structure()
        self.cp_uncertainty = head_structure()

    def forward(self, input_ids, encoder_output, attention_mask=None):
        # Input embedding [batch_size, seq_len, d_model]
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, encoder_output, attention_mask)
            
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Return all property predictions
        return {
            'mw': torch.exp(self.mw_head(x)),
            'pdi': torch.relu(self.pdi_head(x)) + 1.0,
            'phi': torch.sigmoid(self.phi_head(x)),
            'pressure': torch.exp(self.pressure_head(x)),
            'cloud_point': self.cloud_point_head(x),
            'phase': torch.sigmoid(self.phase_head(x)),
            'mw_uncertainty': torch.relu(self.mw_uncertainty(x)),
            'pdi_uncertainty': torch.relu(self.pdi_uncertainty(x)), 
            'phi_uncertainty': torch.relu(self.phi_uncertainty(x)),
            'pressure_uncertainty': torch.relu(self.pressure_uncertainty(x)),
            'cp_uncertainty': torch.relu(self.cp_uncertainty(x))
        }
