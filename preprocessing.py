# polymer_property_predictor/data/utils/preprocessing.py

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
from data_augmentation import PolymerDataAugmentor
from tokenizer_code import ChemicalTokenizer

class PolymerDataProcessor:
    def __init__(self, max_length: int = 150):
        """
        Initialize the data processor with maximum sequence length.
        
        Args:
            max_length: Maximum length for SMILES strings padding
        """
        self.max_length = max_length
        self.atom_vocab = set()
        self.special_tokens = {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'CLS': '[CLS]',
            'SEP': '[SEP]'
        }
        self.normalization_params = {}  # Initialize normalization parameters
        self.augmentor = PolymerDataAugmentor()
        self.column_mappings = {
            'MW(Da)': 'MW (Da)',
            'P(mPa)': 'P (mPa)',
            'CP(°C)': 'CP (°C)',
            'Solvent Smiles': 'Solvent SMILES'
        }
        self.vocab = None  # Initialize vocab as None
        self.target_scaler = RobustScaler()  # Use RobustScaler instead of StandardScaler
        self.target_mean_ = None
        self.target_std_ = None
        
    def process_smiles(self, smiles: str) -> str:
        """
        Process SMILES string to ensure consistency.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Processed SMILES string
        """
        # Handle polymer specific tokens
        smiles = smiles.replace('*', '[*]')  # Handle polymer end groups
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        return Chem.MolToSmiles(mol, canonical=True)
    
    def create_vocabulary(self, smiles_list: List[str]) -> Dict[str, int]:
        """
        Create vocabulary from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary mapping tokens to indices
        """
        # Add special tokens
        vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        
        # Process all SMILES and add unique tokens
        for smiles in smiles_list:
            processed = self.process_smiles(smiles)
            if processed:
                # Split into atoms and bonds
                tokens = []
                current_token = ''
                for char in processed:
                    if char.isupper():
                        if current_token:
                            tokens.append(current_token)
                        current_token = char
                    else:
                        current_token += char
                if current_token:
                    tokens.append(current_token)
                    
                # Add to vocabulary
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)
        
        return vocab
    
    def create_and_assign_vocab(self, smiles_list: List[str]):
        """Create vocabulary from tokenizer"""
        tokenizer = ChemicalTokenizer()
        # Get vocabulary from the tokenizer
        self.vocab = tokenizer.vocab
    
    def normalize_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical properties.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized properties
        """
        props = ['MW (Da)', 'PDI', 'Φ', 'w', 'P (mPa)', 'CP (°C)']
        normalized_df = df.copy()
        
        for prop in props:
            if prop in df.columns:
                # Handle potential non-numeric values
                numeric_mask = pd.to_numeric(df[prop], errors='coerce').notna()
                values = df.loc[numeric_mask, prop].astype(float)
                
                # Add safety check for log transformation
                if prop == 'MW (Da)':
                    values = values.replace(0, np.nan)  # Replace zeros with NaN
                    values = values[values > 0]  # Filter out negative values
                    values = np.log10(values)
                
                # Normalize to [0, 1] range
                min_val = values.min()
                max_val = values.max()
                
                # Convert to float64 before normalization
                normalized_df[prop] = normalized_df[prop].astype(float)
                normalized_df.loc[numeric_mask, prop] = (values - min_val) / (max_val - min_val)
                
                # Store normalization parameters
                self.normalization_params[prop] = {
                    'min': min_val,
                    'max': max_val,
                    'log_transform': prop == 'MW (Da)'
                }
        
        return normalized_df
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
        """
        Prepare dataset for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of input features and target values
        """
        required_cols = ['Polymer SMILES', 'CP(°C)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove outliers and normalize targets
        targets_array = np.array(df['CP(°C)']).reshape(-1, 1)
        if not hasattr(self.target_scaler, 'center_'):
            targets_array = self._remove_outliers(targets_array)
            targets_array = self.target_scaler.fit_transform(targets_array)
        else:
            targets_array = self.target_scaler.transform(targets_array)
        
        # Check for NaNs in targets
        if np.isnan(targets_array).any():
            raise ValueError("NaN detected in target values")
        
        features = {
            'polymer_smiles': df['Polymer SMILES'].tolist()
        }
        targets = {
            'CP(°C)': targets_array.squeeze().tolist()
        }
        
        return features, targets
    
    def _remove_outliers(self, array, k=1.5):
        """Remove outliers using IQR method"""
        q1 = np.percentile(array, 25)
        q3 = np.percentile(array, 75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        return np.clip(array, lower_bound, upper_bound)

    # Add method to inverse transform predictions
    def inverse_transform_targets(self, normalized_values):
        return normalized_values * self.target_std_ + self.target_mean_
    
    def tokenize_and_pad(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Tokenize and pad SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary containing token indices and attention masks
        """
        tokenized = np.zeros((len(smiles_list), self.max_length), dtype=int)
        attention_masks = np.zeros((len(smiles_list), self.max_length), dtype=int)
        pad_idx = self.vocab[self.special_tokens['PAD']]
        
        for i, smiles in enumerate(smiles_list):
            processed = self.process_smiles(smiles)
            if processed:
                tokens = []
                current_token = ''
                for char in processed:
                    if char.isupper():
                        if current_token:
                            tokens.append(current_token)
                        current_token = char
                    else:
                        current_token += char
                if current_token:
                    tokens.append(current_token)
                
                # Convert tokens to indices
                token_indices = [self.vocab.get(token, self.vocab[self.special_tokens['UNK']]) 
                               for token in tokens]
                
                # Add CLS and SEP tokens
                token_indices = [self.vocab[self.special_tokens['CLS']]] + token_indices + [self.vocab[self.special_tokens['SEP']]]
                
                # Pad or truncate
                if len(token_indices) > self.max_length:
                    tokenized[i] = token_indices[:self.max_length]
                    attention_masks[i] = [1] * self.max_length
                else:
                    tokenized[i, :len(token_indices)] = token_indices
                    attention_masks[i, :len(token_indices)] = 1
                    
        return {
            'input_ids': tokenized,
            'attention_mask': attention_masks
        }
    
    def augment_smiles(self, smiles: str, n_augment: int = 5) -> List[str]:
        """
        Augment SMILES string with multiple techniques
        
        Args:
            smiles: Input SMILES string
            n_augment: Number of augmentations to generate
            
        Returns:
            List of augmented SMILES strings
        """
        augmented = []
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return [smiles]
            
        try:
            # Random SMILES generation
            for _ in range(n_augment):
                aug = Chem.MolToSmiles(mol, doRandom=True)
                if aug and aug not in augmented:
                    augmented.append(aug)
                    
            # Chemical substitutions
            subst_aug = self.augmentor.random_substitution(smiles)
            if subst_aug and subst_aug not in augmented:
                augmented.append(subst_aug)
                
        except Exception as e:
            print(f"Error during augmentation: {str(e)}")
            return [smiles]
            
        if not augmented:
            augmented.append(smiles)
            
        return augmented[:n_augment]