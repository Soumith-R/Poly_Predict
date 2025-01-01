import torch
import math
import matplotlib.pyplot as plt

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def range_test(self, train_loader, end_lr=10, num_iter=100):
        """Perform learning rate range test"""
        lr_mult = (end_lr / self.optimizer.param_groups[0]['lr']) ** (1/num_iter)
        best_loss = float('inf')
        avg_loss = 0.
        lr_history = []
        loss_history = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_iter:
                break
                
            # Forward pass
            loss = self._training_step(batch)
            
            # Update learning rate
            self.optimizer.param_groups[0]['lr'] *= lr_mult
            
            # Track best loss and lr
            avg_loss = 0.98 * avg_loss + 0.02 * loss
            smoothed_loss = avg_loss / (1 - 0.98**(batch_idx+1))
            
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                best_lr = self.optimizer.param_groups[0]['lr']
                
            lr_history.append(self.optimizer.param_groups[0]['lr'])
            loss_history.append(smoothed_loss)
            
        return best_lr, lr_history, loss_history
