import torch
import logging

def validate_model_inputs(batch, max_seq_length):
    """Validate model inputs before processing"""
    if batch['input_ids'].dim() != 2:
        raise ValueError(f"Expected input_ids to have 2 dimensions, got {batch['input_ids'].dim()}")
        
    if batch['input_ids'].size(1) > max_seq_length:
        raise ValueError(f"Sequence length {batch['input_ids'].size(1)} exceeds maximum {max_seq_length}")
        
    if not torch.is_floating_point(batch['attention_mask']):
        raise ValueError("attention_mask must be float tensor")
        
    if not batch['input_ids'].dtype == torch.long:
        raise ValueError("input_ids must be long tensor")

def validate_model_outputs(outputs):
    """Validate model outputs"""
    required_keys = ['cloud_point', 'phase', 'encoder_output']
    for key in required_keys:
        if key not in outputs:
            raise ValueError(f"Missing required output key: {key}")
            
    if torch.isnan(outputs['cloud_point']).any():
        raise ValueError("NaN values in cloud_point predictions")
