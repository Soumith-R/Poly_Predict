import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from polymertransformer import PolymerTransformer
from preprocessing import PolymerDataProcessor
from tokenizer_code import ChemicalTokenizer
import numpy as np
import torch.nn as nn
import os
import math
import logging
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from tokenizer_wrapper import TokenizerWrapper
from torch.cuda.amp import GradScaler, autocast  # Update import for mixed precision training

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_dataset(file_path):
    try:
        # Load data
        df = pd.read_excel(file_path)
        print(f"\nLoaded dataset from {file_path}")
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Total samples: {len(df)}")
        
        # Check for required columns
        required_cols = ['Polymer SMILES', 'CP(°C)']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")
        
        # Display data info
        print("\nColumn Info:")
        for col in df.columns:
            non_null = df[col].count()
            print(f"{col}: {non_null} non-null values ({(non_null/len(df))*100:.1f}% complete)")
            
        # Basic data validation
        if df['Polymer SMILES'].isnull().any():
            print("WARNING: Found null values in Polymer SMILES")
        if df['CP(°C)'].isnull().any():
            print("WARNING: Found null values in CP(°C)")
            
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

class PolymerDataset(Dataset):
    def __init__(self, features, targets, tokenizer):
        if not isinstance(features, dict):
            raise ValueError("features must be a dictionary")
        if 'polymer_smiles' not in features:
            raise ValueError("features must contain 'polymer_smiles' key")
        
        self.features = features
        self.targets = targets
        self.tokenizer = TokenizerWrapper(max_length=tokenizer.max_length)

    def __len__(self):
        return len(self.features['polymer_smiles'])

    def __getitem__(self, idx):
        try:
            encoded = self.tokenizer(self.features['polymer_smiles'][idx])
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'targets': torch.tensor(self.targets['CP(°C)'][idx], dtype=torch.float)
            }
        except Exception as e:
            print(f"Error processing item {idx} with text: {self.features['polymer_smiles'][idx]}")
            raise RuntimeError(f"Error processing item {idx}: {str(e)}")

class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() 
        self.huber = nn.SmoothL1Loss()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        huber_loss = self.huber(pred, target)
        return mse_loss + self.alpha * l1_loss + self.beta * huber_loss

def get_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        factor = max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return factor
    
    return LambdaLR(optimizer, lr_lambda)

def remove_weight_norm(model):
    for module in model.modules():
        if hasattr(module, 'weight_g'):
            torch.nn.utils.remove_weight_norm(module)

def train_model(train_data, val_data, model, optimizer, criterion, num_epochs=50, save_dir='checkpoints'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Remove deprecated weight norm
        remove_weight_norm(model)
        
        # Modify warmup and batch settings
        batch_size = 32  # Reduced from 64
        accum_iter = 4   # Reduced from 8
        
        # Increase warmup steps
        total_steps = num_epochs * len(train_data) // batch_size
        warmup_steps = total_steps // 5  # Increased from 10
        
        scheduler = get_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr=1e-6
        )
        
        # Add gradient clipping
        max_grad_norm = 1.0
        
        # Add exponential moving average
        ema = torch.optim.swa_utils.AveragedModel(model)
        
        # Mixed precision training setup
        use_amp = torch.cuda.is_available()  # Only use AMP with CUDA
        scaler = GradScaler() if use_amp else None
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            # Training loop with gradient accumulation
            for i, batch in enumerate(DataLoader(train_data, batch_size=batch_size, shuffle=True)):
                # Convert tensors to correct dtypes
                input_ids = batch['input_ids'].long().to(device)
                attention_mask = batch['attention_mask'].float().to(device)
                targets = batch['targets'].float().to(device)
                
                optimizer.zero_grad()
                
                # Use mixed precision training if available
                if use_amp:
                    with autocast():  # Remove device_type argument
                        outputs = model(
                            src=input_ids,
                            src_padding_mask=attention_mask
                        )
                        loss = criterion(outputs['cloud_point'].squeeze(), targets)
                        loss = loss / accum_iter
                    
                    scaler.scale(loss).backward()
                    
                    if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
                        # Add gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        ema.update_parameters(model)
                else:
                    outputs = model(
                        src=input_ids,
                        src_padding_mask=attention_mask
                    )
                    loss = criterion(outputs['cloud_point'].squeeze(), targets)
                    loss = loss / accum_iter
                    loss.backward()
                    
                    if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
                        optimizer.step()
                        optimizer.zero_grad()
                        ema.update_parameters(model)
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_data)
            
            # Validation with correct dtypes
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in DataLoader(val_data, batch_size=32):
                    input_ids = batch['input_ids'].long().to(device)
                    attention_mask = batch['attention_mask'].float().to(device)
                    targets = batch['targets'].float().to(device)
                    
                    # Use model's forward method directly
                    outputs = model(
                        src=input_ids,
                        src_padding_mask=attention_mask
                    )
                    val_loss += criterion(outputs['cloud_point'].squeeze(), targets).item()
            
            avg_val_loss = val_loss / len(val_data)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'scaler': scaler.state_dict(),  # Save scaler state
                }
                torch.save(checkpoint, f'{save_dir}/model_best.pt')
                print(f"Model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    except RuntimeError as e:
        if "out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.error("GPU out of memory. Try reducing batch size.")
        raise
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tokenizer
        tokenizer = ChemicalTokenizer(max_length=150)
        
        # Load and validate data
        data_processor = PolymerDataProcessor()
        train_df = load_dataset('Polymer6kDataset.xlsx')
        
        if train_df is None or len(train_df) == 0:
            raise ValueError("Failed to load training data or empty dataset")
        
        # Create and assign vocabulary
        data_processor.create_and_assign_vocab(train_df['Polymer SMILES'].tolist())
        
        # Split into train/val with error handling
        try:
            train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
            
            # Process datasets with validation
            train_features, train_targets = data_processor.prepare_dataset(train_df)
            val_features, val_targets = data_processor.prepare_dataset(val_df)
            
            # Create datasets with proper error handling
            train_data = PolymerDataset(train_features, train_targets, tokenizer)
            val_data = PolymerDataset(val_features, val_targets, tokenizer)
            
        except Exception as e:
            raise RuntimeError(f"Error preparing datasets: {str(e)}")
        
        # Initialize with even smaller model for initial training
        model = PolymerTransformer(
            vocab_size=len(data_processor.vocab),
            d_model=256,        # Further reduced
            n_layers=6,         # Further reduced
            n_heads=8,
            d_ff=1024,         # Further reduced
            dropout=0.1,        # Reduced dropout initially
            max_seq_length=150
        )
        
        # Modified optimizer settings
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,          # Increased from 1e-4
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Add learning rate warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=50,
            steps_per_epoch=len(train_data) // 32,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Use simple MSE loss
        criterion = nn.MSELoss()

        # Initialize simple GradScaler without device_type
        scaler = GradScaler()

        # Reduce number of epochs and add more patience
        train_model(
            train_data, 
            val_data, 
            model, 
            optimizer, 
            criterion, 
            num_epochs=50,     # Reduced epochs
            save_dir=save_dir
        )
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise