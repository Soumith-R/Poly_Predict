import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model loaded from {path}')

def get_model_architecture(state_dict):
    """Extract model architecture parameters from state dict"""
    # Count number of decoder layers
    n_layers = max([int(key.split('.')[2]) for key in state_dict.keys() 
                   if key.startswith('decoder.layers')]) + 1
    
    # Get d_model from embedding dimension
    d_model = state_dict['shared_embedding.weight'].shape[1]
    
    # Get n_heads from attention weights shape
    # Using first layer's attention projection matrix
    attn_weight = state_dict['encoder.layers.0.self_attention.in_proj_weight']
    n_heads = attn_weight.shape[0] // (3 * d_model)
    
    return {
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'vocab_size': state_dict['shared_embedding.weight'].shape[0]
    }

def print_model_info(checkpoint_path):
    """Load checkpoint and print architecture info"""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Extract architecture
        arch_info = get_model_architecture(state_dict)
        
        print("\nModel Architecture:")
        print(f"Vocab Size: {arch_info['vocab_size']}")
        print(f"d_model: {arch_info['d_model']}")
        print(f"n_layers: {arch_info['n_layers']}")
        print(f"n_heads: {arch_info['n_heads']}")
        
        return arch_info
        
    except Exception as e:
        print(f"Error analyzing checkpoint: {str(e)}")
        return None