import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        
    def update(self, loss, preds, targets):
        self.losses.append(loss)
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        
    def compute(self):
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return {
            'loss': np.mean(self.losses),
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }
        
    def get_current_metrics(self):
        metrics = self.compute()
        return (f"Loss: {metrics['loss']:.4f}, "
                f"MSE: {metrics['mse']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}, "
                f"MAE: {metrics['mae']:.4f}, "
                f"R2: {metrics['r2']:.4f}")
