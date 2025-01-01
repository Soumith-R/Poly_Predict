import torch
import logging
from polymertransformer import PolymerDecoder

logging.basicConfig(level=logging.INFO)

def test_saved_model(test_data_path, checkpoint_path, batch_size=16):
    model = PolymerDecoder()  # Assuming PolymerDecoder is defined elsewhere
    try:
        model.load_state_dict(torch.load(checkpoint_path))
    except RuntimeError as e:
        logging.error(f"Error loading model state_dict: {e}")
        return None, None

    # Continue with testing logic...
    predictions = []  # Placeholder for predictions
    evaluation = {}   # Placeholder for evaluation metrics
    return predictions, evaluation

if __name__ == "__main__":
    test_data_path = "Polymer6kDataset.xlsx"
    checkpoint_path = "checkpoints/model_best.pt"
    predictions, evaluation = test_saved_model(test_data_path, checkpoint_path)