import torch
import pandas as pd
import logging
import os
from typing import Dict, Union, List
import psutil
import warnings
from transformerdecoder import PolymerDecoder  # Corrected import
from preprocessing import PolymerDataProcessor  # Corrected import
from rdkit import Chem  # Add missing import for SMILES validation
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class PolymerPredictor:
    def __init__(self, model_path: str = 'checkpoints/model_best.pt'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = None
        self.model = None
        self.property_ranges = {
            'MW (Da)': (100, 10000),
            'PDI': (1.0, 3.0),
            'Φ': (0.0, 1.0), 
            'P (mPa)': (0.1, 1000.0),
            'CP (°C)': (-50, 150),
            'Phase Stability': (0, 1)
        }
        
    def _load_model(self):
        """Load model with memory optimization for CPU"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            # Load checkpoint with explicit weights_only=True
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=True  # Safe loading
            )
            
            # Get model configuration
            vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
            
            # Initialize model
            self.model = PolymerDecoder(
                vocab_size=vocab_size,
                d_model=768,
                n_layers=12,
                n_heads=12,
                d_ff=3072,
                dropout=0.1
            ).to(self.device)
            
            if torch.cuda.is_available():
                self.model = self.model.half()
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Clear memory
            del checkpoint
            torch.cuda.empty_cache()
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    @torch.inference_mode()
    def predict_single(self, smile_string: str) -> Dict[str, Union[float, str]]:
        """Make predictions for all polymer properties"""
        if not isinstance(smile_string, str) or not smile_string.strip():
            raise ValueError("Invalid SMILE string")

        try:
            # Lazy loading
            if self.model is None:
                self._load_model()
            if self.data_processor is None:
                self.data_processor = PolymerDataProcessor()

            # Create input features with all required columns
            input_df = pd.DataFrame({
                'Polymer SMILES': [smile_string],
                'Solvent SMILES': [''],
                'MW (Da)': [1000.0],
                'PDI': [1.0],
                'Φ': [0.0],
                'P (mPa)': [1.0],
                'CP (°C)': [0.0],
                '1-Phase': ['positive']  # Added missing column
            })

            # Process input
            features, _ = self.data_processor.prepare_dataset(input_df)
            
            # Convert to tensors
            input_ids = torch.tensor(
                features['polymer_smiles'], 
                dtype=torch.long, 
                device=self.device
            )
            attention_mask = torch.tensor(
                features['attention_mask'], 
                dtype=torch.float, 
                device=self.device
            )
            
            # Create encoder output
            encoder_output = torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.model.d_model,
                device=self.device,
                dtype=torch.float32
            )

            # Get all predictions
            outputs = self.model(input_ids, encoder_output, attention_mask)
            
            # Extract all property predictions
            predictions = {
                'SMILE': smile_string,
                'Molecular Weight': {
                    'Value': outputs['mw'].squeeze().item(),
                    'Unit': 'Da',
                    'Range': self._get_confidence_interval(outputs['mw'].squeeze().item(), 0.1),
                    'Confidence': self._get_prediction_confidence(outputs['mw_uncertainty'].squeeze().item())
                },
                'Polydispersity Index': {
                    'Value': outputs['pdi'].squeeze().item(),
                    'Range': self._get_confidence_interval(outputs['pdi'].squeeze().item(), 0.05),
                    'Confidence': self._get_prediction_confidence(outputs['pdi_uncertainty'].squeeze().item())
                },
                'Volume Fraction': {
                    'Value': outputs['phi'].squeeze().item(),
                    'Range': self._get_confidence_interval(outputs['phi'].squeeze().item(), 0.02),
                    'Confidence': self._get_prediction_confidence(outputs['phi_uncertainty'].squeeze().item())
                },
                'Pressure': {
                    'Value': outputs['pressure'].squeeze().item(),
                    'Unit': 'mPa',
                    'Range': self._get_confidence_interval(outputs['pressure'].squeeze().item(), 0.1),
                    'Confidence': self._get_prediction_confidence(outputs['pressure_uncertainty'].squeeze().item())
                },
                'Cloud Point': {
                    'Value': outputs['cloud_point'].squeeze().item(),
                    'Unit': '°C',
                    'Range': self._get_confidence_interval(outputs['cloud_point'].squeeze().item(), 5),
                    'Confidence': self._get_prediction_confidence(outputs['cp_uncertainty'].squeeze().item())
                },
                'Phase': {
                    'Value': 'Two-Phase' if outputs['phase'].squeeze().item() < 0.5 else 'One-Phase',
                    'Probability': outputs['phase'].squeeze().item(),
                    'Confidence': f"{abs(outputs['phase'].squeeze().item() - 0.5) * 200:.1f}%"
                },
                'Overall Reliability': self._calculate_overall_reliability(outputs)
            }

            # Clean memory
            del features, input_ids, attention_mask, encoder_output
            
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
            outputs['mw_uncertainty'].squeeze().item(),
            outputs['pdi_uncertainty'].squeeze().item(),
            outputs['phi_uncertainty'].squeeze().item(),
            outputs['pressure_uncertainty'].squeeze().item(),
            outputs['cp_uncertainty'].squeeze().item(),
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
    min_memory_gb = 1.0
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    logging.info(f"Available memory: {available_memory:.2f}GB")
    
    if available_memory < min_memory_gb:
        logging.warning(f"Low memory! Minimum {min_memory_gb}GB recommended. Available: {available_memory:.2f}GB")
        return False
    return True

def main():
    try:
        # Check system requirements
        if not check_system_requirements():
            logging.error("System requirements not met")
            return

        # Get SMILE string from user
        smile_string = input("Please enter the Polymer SMILE string: ").strip()
        
        if not smile_string:
            logging.error("Empty SMILE string provided")
            return

        predictor = PolymerPredictor()
        
        try:
            result = predictor.predict_single(smile_string)
            print("\nPrediction Results:")
            print(f"Input SMILE: {result['SMILE']}")
            print("-" * 50)
            
            # Print all property predictions
            for prop, data in result.items():
                if prop not in ['SMILE', 'Overall Reliability']:
                    print(f"\n{prop}:")
                    for key, value in data.items():
                        print(f"  {key}: {value}")
            
            print(f"\nOverall Reliability: {result['Overall Reliability']}")
            print("\nNote: Confidence < 50% indicates high uncertainty")
                    
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            
    finally:
        # Clean up
        if 'predictor' in locals():
            del predictor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
