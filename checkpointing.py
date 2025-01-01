import torch
import os
import json

class ModelCheckpointing:
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_score = float('inf')
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, metrics):
        """Save model checkpoint with metrics"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'model_config': model.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, f'{self.save_dir}/latest_{self.model_name}.pt')
        
        # Save best model
        if loss < self.best_score:
            self.best_score = loss
            torch.save(checkpoint, f'{self.save_dir}/best_{self.model_name}.pt')
            
            # Save metrics
            with open(f'{self.save_dir}/best_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
