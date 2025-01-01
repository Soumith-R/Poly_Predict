# test_train.py
import torch
from polymertransformer import PolymerTransformer  # Updated import
from torch.utils.data import DataLoader
from tokenizer_code import ChemicalTokenizer
from transformerdecoder import PolymerDecoder
from preprocessing import PolymerDataProcessor
from data_augmentation import PolymerDataAugmentor
from train import PolymerDataset
import pandas as pd


def test_train_model():
    # Create test data with correct column names
    sample_data = {
        'Polymer SMILES': ['CC(=O)OC1=CC=CC=C1C(=C)C(=O)O'],
        'Solvent SMILES': ['C(C)C'],
        'MW(Da)': [100.0],  # Fixed column name
        'PDI': [1.0],
        'Φ': [0.5],
        'P(mPa)': [100.0],  # Fixed column name  
        'CP(°C)': [50.0],   # Fixed column name
        '1-Phase': ['positive']
    }
    test_dataframe = pd.DataFrame(sample_data)

    # Initialize data processor
    data_processor = PolymerDataProcessor()
    features, targets = data_processor.prepare_dataset(test_dataframe)

    # Create test dataset
    tokenizer = ChemicalTokenizer(max_length=150)
    test_dataset = PolymerDataset(features, targets, tokenizer)

    # Initialize model with minimal config for testing
    model = PolymerDecoder(
        vocab_size=len(data_processor.vocab),
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )

    # Run a test forward pass
    batch = next(iter(DataLoader(test_dataset, batch_size=1)))
    outputs = model(
        input_ids=batch['input_ids'],
        encoder_output=torch.randn(1, 12, 768),
        attention_mask=batch['attention_mask']
    )

    # Basic assertion
    assert 'cloud_point' in outputs
    assert outputs['cloud_point'].shape == (1, 1)

if __name__ == '__main__':
    test_train_model()
