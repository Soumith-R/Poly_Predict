import torch
import gc
import psutil
import os

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved()
        }
    return None

def optimize_memory_usage(model, batch_size, seq_length):
    """Optimize memory usage based on model and input size"""
    try:
        # Estimate memory requirements
        sample_input = torch.randint(0, 100, (batch_size, seq_length))
        torch.cuda.empty_cache()
        gc.collect()
        
        # Test forward pass
        with torch.no_grad():
            _ = model(sample_input)
            
        mem_info = get_gpu_memory_info()
        if mem_info:
            required_memory = mem_info['allocated']
            available_memory = torch.cuda.get_device_properties(0).total_memory
            
            if required_memory > 0.9 * available_memory:
                recommended_batch = int(batch_size * 0.8)
                return {
                    'status': 'warning',
                    'message': f'Memory usage high. Consider reducing batch size to {recommended_batch}'
                }
        
        return {'status': 'ok'}
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return {
                'status': 'error',
                'message': 'Not enough GPU memory. Try reducing batch size or sequence length.'
            }
        raise

def enable_memory_efficient_training(model):
    """Enable memory efficient training features"""
    # Enable gradient checkpointing
    if hasattr(model, 'encoder'):
        model.encoder.gradient_checkpointing_enable()
    if hasattr(model, 'decoder'):
        model.decoder.gradient_checkpointing_enable()
    
    # Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    return scaler
