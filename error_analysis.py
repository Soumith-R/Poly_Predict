import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_predictions(true_values, predictions, save_dir='results'):
    """Analyze prediction errors and generate visualizations"""
    
    # Calculate residuals
    residuals = predictions - true_values
    
    # Plot residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Prediction Residuals Distribution')
    plt.savefig(f'{save_dir}/residuals.png')
    
    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.savefig(f'{save_dir}/predictions_vs_true.png')
    
    # Calculate error statistics
    error_stats = {
        'mean_error': np.mean(residuals),
        'std_error': np.std(residuals),
        'max_error': np.max(np.abs(residuals)),
        'median_error': np.median(np.abs(residuals))
    }
    
    return error_stats
