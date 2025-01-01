import torch
import pandas as pd
from polymertransformer import PolymerTransformer
from preprocessing import PolymerDataProcessor
from tokenizer_code import ChemicalTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt  # Add matplotlib for plotting

def load_test_data(file_path):
    """Load and validate test dataset"""
    df = pd.read_excel(file_path)
    
    # Ensure column names match exactly
    if 'CP(°C)' not in df.columns and 'CP (°C)' in df.columns:
        df = df.rename(columns={'CP (°C)': 'CP(°C)'})
    
    # Validate required columns
    required_cols = ['Polymer SMILES', 'CP(°C)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in test data: {missing_cols}")
    
    return df

def convert_state_dict(state_dict):
    """Convert weight normalized parameters to regular parameters"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if '_g' in key or '_v' in key:
            # Skip weight norm parameters
            continue
        if 'weight' in key and f"{key}_g" in state_dict:
            # Combine weight norm parameters
            g = state_dict[f"{key}_g"]
            v = state_dict[f"{key}_v"]
            new_state_dict[key] = v * (g / torch.norm(v))
        else:
            new_state_dict[key] = value
    return new_state_dict

class TestDataset(Dataset):
    def __init__(self, features, tokenizer):
        self.features = features
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.features['polymer_smiles'])

    def __getitem__(self, idx):
        try:
            encoded = self.tokenizer.encode(
                self.features['polymer_smiles'][idx],
                add_special_tokens=True
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing item {idx} with text: {self.features['polymer_smiles'][idx]}")
            raise RuntimeError(f"Error processing item {idx}: {str(e)}")

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

def plot_predictions_vs_actuals(predictions, actuals):
    """Plot predictions vs. actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--')
    plt.xlabel('Actual CP(°C)')
    plt.ylabel('Predicted CP(°C)')
    plt.title('Predicted vs. Actual CP(°C)')
    plt.grid(True)
    plt.show()

def test_saved_model(test_data_path, checkpoint_path, batch_size=16):
    """
    Test a saved model on new data.
    
    Args:
        test_data_path: Path to test dataset
        checkpoint_path: Path to saved model checkpoint
        batch_size: Batch size for testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_df = load_test_data(test_data_path)
    
    # Initialize tokenizer and data processor
    tokenizer = ChemicalTokenizer(max_length=150)
    data_processor = PolymerDataProcessor()
    
    # Create vocabulary from test data
    data_processor.create_and_assign_vocab(test_df['Polymer SMILES'].tolist())
    
    # Prepare test dataset
    test_features, test_targets = data_processor.prepare_dataset(test_df)
    
    # Create test dataset with proper handling
    test_dataset = TestDataset(test_features, tokenizer)
    
    # Load checkpoint with safety flag
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Initialize model with the same configuration as the saved checkpoint
    model = PolymerTransformer(
        vocab_size=len(data_processor.vocab),
        d_model=512,        # Match saved checkpoint configuration
        n_layers=12,        # Match saved checkpoint configuration
        n_heads=12,         # Match saved checkpoint configuration
        d_ff=2048,          # Match saved checkpoint configuration
        dropout=0.2,        # Match saved checkpoint configuration
        max_seq_length=150
    )
    
    # Convert and load state dict
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
    
    # Remove any remaining weight norm
    for module in model.modules():
        if hasattr(module, 'weight_g'):
            delattr(module, 'weight_g')
            delattr(module, 'weight_v')
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                src=input_ids,
                src_padding_mask=attention_mask
            )
            
            pred = outputs['cloud_point'].squeeze().cpu().numpy()
            if pred.ndim == 0:  # Handle single prediction
                pred = [pred.item()]
            predictions.extend(pred)
    
    actuals = test_df['CP(°C)'].tolist()
    
    # Ensure predictions and actuals have the same length
    if len(predictions) > len(actuals):
        predictions = predictions[:len(actuals)]
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Convert predictions and metrics to serializable format
    predictions = convert_to_serializable(predictions)
    actuals = convert_to_serializable(actuals)
    
    evaluation = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    # Save results
    results = {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': evaluation
    }
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save with proper conversion
    with open('results/test_results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=4)
    
    # Plot predictions vs. actuals
    plot_predictions_vs_actuals(predictions, actuals)
    
    return predictions, evaluation

if __name__ == "__main__":
    test_data_path = 'Polymer6kDataset.xlsx'  # Update with your test data path
    checkpoint_path = 'checkpoints/model_best.pt'  # Update with your checkpoint path
    
    try:
        predictions, evaluation = test_saved_model(test_data_path, checkpoint_path)
        print("\nTest Results:")
        print(f"MSE: {evaluation['mse']:.4f}")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"R2 Score: {evaluation['r2']:.4f}")
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        raise