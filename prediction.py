import torch
import pandas as pd
import logging
import os
from typing import Dict, Union, List
import psutil
import warnings
from polymertransformer import PolymerTransformer  # Import the main model
from preprocessing import PolymerDataProcessor  # Corrected import
from tokenizer_code import ChemicalTokenizer  # Corrected import
from rdkit import Chem  # Add missing import for SMILES validation
import math
from transformerdecoder import PolymerDecoder  # At top of file
from torch import nn

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging to write to file instead of terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'polymer_prediction.log')),
        logging.StreamHandler(open(os.devnull, 'w'))  # Suppress terminal output
    ]
)

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class Linear1D(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.bias = nn.Parameter(torch.zeros(size))
    
    def forward(self, x):
        return x * self.weight + self.bias

class PolymerPredictor:
    def __init__(self, model_path: str = "checkpoints/model_best.pt", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.abspath(model_path)
        self.model = None
        # Don't initialize vocab here, load from checkpoint instead
        self.vocab = None
        
    def _initialize_vocab(self, checkpoint_vocab_size):
        """Initialize vocabulary to match checkpoint exactly"""
        chars = "C[]()+=#-1234567890.OHN"
        
        logging.debug(f"Chars string: {chars}")
        logging.debug(f"Chars length: {len(chars)}")
        logging.debug(f"Expected vocab size: {checkpoint_vocab_size}")
        logging.debug(f"Target chars length should be: {checkpoint_vocab_size-1}")
        
        if len(chars) != checkpoint_vocab_size - 1:
            raise ValueError(
                f"Vocab size mismatch. Need exactly {checkpoint_vocab_size-1} chars, got {len(chars)}."
            )
        
        vocab = {c: i+1 for i, c in enumerate(chars)}
        vocab['[PAD]'] = 0
        
        logging.debug(f"Final vocab size: {len(vocab)}")
        return vocab

    def _load_model(self):
        try:
            # 1. Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint format")
            state_dict = checkpoint['model_state_dict']

            # 2. Initialize vocabulary first
            vocab_size = 24  # Fixed vocab size from architecture
            self.vocab = self._initialize_vocab(vocab_size)
            
            # 3. Initialize model
            self.model = PolymerDecoder(
                vocab_size=vocab_size,
                d_model=512,
                n_layers=12,
                n_heads=8,
                d_ff=2048,
                dropout=0.1
            ).to(self.device)

            # 3. Initialize missing embeddings
            self.model.pos_embedding = nn.Parameter(
                torch.zeros(1, 512, 512)  # [1, max_seq_len, d_model]
            ).to(self.device)
            
            self.model.token_embedding = nn.Embedding(
                24, 512,  # vocab_size, d_model
                padding_idx=0
            ).to(self.device)

            # 4. Update prediction heads
            self.model.cloud_point_head = nn.Sequential(
                Linear1D(512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            )

            self.model.phase_head = nn.Sequential(
                Linear1D(512),
                nn.GELU(), 
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

            # 5. Load state dict with strict=False to handle missing keys
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"Missing keys: {missing}")
            if unexpected:  
                logging.warning(f"Unexpected keys: {unexpected}")

            self.model.eval()
            logging.info("Model loaded successfully")

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    @torch.inference_mode()
    def predict_single(self, smile_string: str):
        try:
            if self.model is None:
                self._load_model()
                
            # Ensure fixed sequence length
            max_seq_length = 512
            input_str = (smile_string + '[PAD]' * max_seq_length)[:max_seq_length]
                
            # Create input tensors 
            input_ids = torch.tensor(
                [[self.vocab.get(c, 0) for c in input_str]], 
                dtype=torch.long,
                device=self.device
            )
            
            attention_mask = torch.ones(
                (1, max_seq_length),
                dtype=torch.float, 
                device=self.device
            )
            attention_mask[0, len(smile_string):] = 0
            
            # Match encoder output dimensions
            encoder_output = torch.zeros(
                1,              # batch size
                max_seq_length, # sequence length  
                512,           # d_model
                device=self.device,
                dtype=torch.float
            )

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                encoder_output=encoder_output,
                attention_mask=attention_mask
            )

            # Extract predictions with proper formatting
            predictions = {
                'SMILE': smile_string,
                'Molecular Weight': {
                    'Value': outputs['mw'].squeeze().item(),
                    'Unit': 'Da',
                    'Uncertainty': outputs['mw_uncertainty'].squeeze().item()
                },
                'Polydispersity Index': {
                    'Value': outputs['pdi'].squeeze().item(),
                    'Uncertainty': outputs['pdi_uncertainty'].squeeze().item()
                },
                'Volume Fraction': {
                    'Value': outputs['phi'].squeeze().item(),
                    'Uncertainty': outputs['phi_uncertainty'].squeeze().item()
                },
                'Pressure': {
                    'Value': outputs['pressure'].squeeze().item(),
                    'Unit': 'mPa',
                    'Uncertainty': outputs['pressure_uncertainty'].squeeze().item() 
                },
                'Cloud Point': {
                    'Value': outputs['cloud_point'].squeeze().item(),
                    'Unit': '°C',
                    'Uncertainty': outputs['cp_uncertainty'].squeeze().item()
                },
                'Phase': {
                    'Value': 'Two-Phase' if outputs['phase'].squeeze().item() < 0.5 else 'One-Phase',
                    'Probability': f"{abs(0.5 - outputs['phase'].squeeze().item()) * 200:.1f}%"
                }
            }

            return predictions

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

    def _get_confidence_interval(self, value: float, margin_percent: float) -> str:
        """Calculate confidence interval for a predicted value"""
        margin = abs(value) * margin_percent
        return f"{value-margin:.2f} to {value+margin:.2f}"

    def _get_prediction_confidence(self, uncertainty: float) -> str:
        """Convert uncertainty to confidence percentage"""
        confidence = (1 - min(uncertainty, 1)) * 100
        return f"{confidence:.1f}%"

    def _calculate_overall_reliability(self, outputs: Dict) -> str:
        """Calculate overall prediction reliability"""
        uncertainties = [
            outputs['cloud_point'].squeeze().item(),
            outputs['phase'].squeeze().item(),
            outputs['phase'].squeeze().item(),
            outputs['phase'].squeeze().item(),
            outputs['phase'].squeeze().item(),
            abs(0.5 - outputs['phase'].squeeze().item())
        ]
        reliability = (1 - sum(uncertainties) / len(uncertainties)) * 100
        return f"{max(0, min(100, reliability)):.1f}%"

    # Add to PolymerPredictor
    def calibrate_predictions(self, predictions):
        """Apply temperature scaling to calibrate confidence"""
        temperature = 1.5  # Tune this value
        for key in predictions:
            if isinstance(predictions[key], dict) and 'Confidence' in predictions[key]:
                conf = float(predictions[key]['Confidence'].strip('%')) / 100
                calibrated_conf = torch.sigmoid(torch.tensor(conf) / temperature).item()
                predictions[key]['Confidence'] = f"{calibrated_conf*100:.1f}%"
        return predictions

    # Add to PolymerPredictor
    def _validate_smile(self, smile_string):
        """Validate and normalize SMILE string"""
        try:
            mol = Chem.MolFromSmiles(smile_string)
            if mol is None:
                raise ValueError("Invalid SMILE string")
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            raise ValueError(f"SMILE validation failed: {str(e)}")

class EnsemblePredictor:
    def __init__(self, model_paths):
        self.models = [PolymerPredictor(path) for path in model_paths]
    
    def predict(self, smile_string):
        predictions = []
        for model in self.models:
            pred = model.predict_single(smile_string)
            predictions.append(pred)
        return self._aggregate_predictions(predictions)

def check_system_requirements():
    """Check if system meets minimum requirements"""
    min_memory_gb = 0.5
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    logging.info(f"Available memory: {available_memory:.2f}GB")
    
    if available_memory < min_memory_gb:
        logging.warning(f"Low memory! Minimum {min_memory_gb}GB recommended. Available: {available_memory:.2f}GB")
        return False
    return True

# Add column name validation function
def validate_column_names(df):
    """Validate and fix column names if needed"""
    column_mappings = {
        'MW (Da)': 'MW(Da)',
        'P (mPa)': 'P(mPa)', 
        'CP (°C)': 'CP(°C)',
        'Solvent Smiles': 'Solvent SMILES'
    }
    
    # Rename columns if needed
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            
    return df

def main():
    try:
        # Check system requirements
        if not check_system_requirements():
            logging.error("System requirements not met")
            return

        # Show example SMILES
        print("\nExample valid SMILES: CC(=O)OC1=CC=CC=C1")
        print("Please enter a valid polymer SMILES string.")
        
        # Get SMILE string from user with validation
        while True:
            smile_string = input("\nPolymer SMILE string: ").strip()
            
            if not smile_string:
                print("Empty input. Please enter a SMILES string.")
                continue
                
            # Basic SMILES validation
            if not all(c in "C[]()+=#-1234567890.OHNPS" for c in smile_string):
                print("Invalid SMILES format. Please use only valid chemical symbols and notation.")
                print("Valid characters are: C, H, O, N, P, S and []()+=# -1234567890.")
                continue
                
            break

        predictor = PolymerPredictor()
        
        try:
            result = predictor.predict_single(smile_string)
            print("\nPrediction Results:")
            print(f"Input SMILE: {result['SMILE']}")
            print("-" * 50)

            for prop, data in result.items():
                if prop != 'SMILE':
                    print(f"\n{prop}:")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {data}")
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            
    finally:
        # Clean up
        if 'predictor' in locals():
            del predictor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
