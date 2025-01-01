import torch
import logging
from transformerdecoder import PolymerDecoder

def load_model_state(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError as e:
        logging.error("Error loading model state_dict: %s", e)
        logging.error("Ensure that the model architecture matches the checkpoint.")
        raise

def test_load_model():
    model = PolymerDecoder(
        vocab_size=100,  # Size of your SMILES vocabulary
        d_model=768,     # Model dimension
        n_layers=12,     # Number of decoder layers
        n_heads=12,      # Number of attention heads
        d_ff=3072,       # Feed-forward dimension
        dropout=0.1      # Dropout rate
    )
    checkpoint_path = 'checkpoints/model_best.pt'
    load_model_state(model, checkpoint_path)

if __name__ == "__main__":
    test_load_model()