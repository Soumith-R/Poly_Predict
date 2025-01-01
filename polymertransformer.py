import torch
import torch.nn as nn
from transformerencoder import PolymerEncoder  # Local import
from transformerdecoder import PolymerDecoder  # Local import

class PolymerTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model=512,  # Match checkpoint dimension
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=150
    ):
        super().__init__()
        
        # Create encoder and decoder with shared embedding
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        self.encoder = PolymerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Update prediction heads to use Sequential layers
        head_structure = lambda: nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.cloud_point_head = head_structure()
        self.phase_head = head_structure()
        self.mw_head = head_structure()
        self.pdi_head = head_structure()
        self.phi_head = head_structure()
        self.pressure_head = head_structure()
        
        # Update uncertainty heads
        self.mw_uncertainty = head_structure()
        self.pdi_uncertainty = head_structure()
        self.phi_uncertainty = head_structure()
        self.pressure_uncertainty = head_structure()
        self.cp_uncertainty = head_structure()

        # Initialize parameters
        self._init_parameters()

        # Store model configuration
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'dropout': dropout,
            'max_seq_length': max_seq_length
        }

    def _init_parameters(self):
        """Initialize or reset model parameters"""
        # Initialize embedding
        nn.init.normal_(self.shared_embedding.weight, mean=0, std=0.02)
        
        # Initialize heads
        for head in [self.cloud_point_head, self.phase_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, src, src_padding_mask=None):
        if src.dim() != 2:
            raise ValueError(f"Expected src to have 2 dimensions, got {src.dim()}")
            
        try:
            # Get encoder output and reshape
            encoder_output = self.encoder(src, src_padding_mask)[0]  # [batch_size, seq_len, d_model]
            
            # Add extra dimension and transpose
            encoder_output = encoder_output.transpose(0, 1).view(encoder_output.size(1), -1)  # [seq_len, batch_size * d_model]
            encoder_output = encoder_output.view(encoder_output.size(0), -1, 768)  # [seq_len, batch_size, d_model]
            
            # Global average pooling
            pooled_output = encoder_output.mean(dim=1)
            
            # Return predictions dictionary
            return {
                'mw': self.mw_head(pooled_output),
                'pdi': self.pdi_head(pooled_output),
                'phi': self.phi_head(pooled_output),
                'pressure': self.pressure_head(pooled_output),
                'cloud_point': self.cloud_point_head(pooled_output),
                'phase': torch.sigmoid(self.phase_head(pooled_output)),
                'mw_uncertainty': torch.relu(self.mw_uncertainty(pooled_output)),
                'pdi_uncertainty': torch.relu(self.pdi_uncertainty(pooled_output)),
                'phi_uncertainty': torch.relu(self.phi_uncertainty(pooled_output)),
                'pressure_uncertainty': torch.relu(self.pressure_uncertainty(pooled_output)),
                'cp_uncertainty': torch.relu(self.cp_uncertainty(pooled_output))
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(f"GPU OOM: Try reducing batch size or sequence length. Current shape: {src.shape}")
            raise

    def get_attention_weights(self):
        """Get attention weights for model interpretability"""
        return {
            'encoder_attention': [layer.self_attention.attn_weights for layer in self.encoder.layers],
            'decoder_attention': [layer.self_attention.attn_weights for layer in self.decoder.layers]
        }

    def load_pretrained(self, path):
        """Load pretrained weights with validation"""
        state_dict = torch.load(path)
        if not self._validate_state_dict(state_dict):
            raise ValueError("Invalid state dict for model configuration")
        self.load_state_dict(state_dict)

    def _validate_state_dict(self, state_dict):
        """Validate loaded state dict against current model configuration"""
        try:
            expected_keys = set(self.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            if expected_keys != loaded_keys:
                missing = expected_keys - loaded_keys
                extra = loaded_keys - expected_keys
                raise ValueError(f"Mismatched keys. Missing: {missing}, Extra: {extra}")
            return True
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def reset_parameters(self):
        """Reset all parameters"""
        self._init_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
