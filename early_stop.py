import numpy as np
from loggs import logger
import os
import torch

class Early_Stopping:
    """
    Early stopping class to automatically stop training at given conditions
    """
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """ 
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        
        self.patience = patience         # Number of epochs to wait after the last improvement
        self.verbose = verbose           # If True, print messages for improvements
        self.counter = 0                 # Counts the number of epochs since the last improvement
        self.best_score = None           # Best score observed so far (negative validation loss)
        self.early_stop = False          # Flag to indicate whether to stop early
        self.val_loss_min = np.Inf       # Minimum validation loss observed (initially set to infinity)
        self.delta = delta               # Minimum improvement required to reset the patience counter
        self.best_epoch = 0              # The epoch at which the best model was observed


    def __call__(self, val_loss, epoch):
        """
        Call this method at the end of each epoch to determine whether early stopping should occur.
        Args:
            val_loss (float): The current epoch's validation loss.
            epoch (int): The current epoch number.
        """
        
        current_score = -val_loss                                 # Negate the validation loss because we're looking for a decrease
        if self.best_score is None:                               # If no best score has been recorded yet
            self.best_score = current_score                       # Set the first observed score as the best
            self.val_loss_min = val_loss                          # Set the minimum validation loss to the current one
            self.best_epoch = epoch                               # Record the epoch number
        elif current_score < self.best_score + self.delta:        # If no significant improvement
            self.counter += 1                                     # Increment the counter
            if self.counter >= self.patience:                     # If patience is exceeded
                self.early_stop = True                            # Trigger early stopping
                if self.verbose:                                  # If verbose is enabled
                    logger.info(f"Early stopping triggered at epoch {epoch}. Patience of {self.patience} epochs reached without improvement greater than {self.delta}.")
        else:                                                # If there is an improvement
            self.best_score = current_score                  # Update the best score
            self.val_loss_min = val_loss                     # Update the minimum validation loss
            self.best_epoch = epoch                          # Update the best epoch
            self.counter = 0                                 # Reset the patience counter
            
    def load_best_model(self, model, save_model_path, model_name):
        """
        Load the best saved model state.
        Args:
            model (torch.nn.Module): The model instance to load the state into.
            save_model_path (str): Directory where the model was saved.
            model_name (str): The base name of the model to load.
        """
        best_model_path = os.path.join(save_model_path, f"{model_name}_best_epoch_{self.best_epoch}.pt")
        model.load_state_dict(torch.load(best_model_path))      # Load the model state from the saved file
        if self.verbose:
            logger.info(f"Loaded best model from {best_model_path}")
            
            
### Example
#early_stopping = Early_Stopping(patience=5, verbose=True, delta=0.0001)
# for epoch in range(epochs):
#     train(...)
#     val_loss = validate(...)
#     early_stopping(val_loss, model)
#     if early_stopping.early_stop:
#         self.logger("Stopping training early")
#         early_stopping.load_best_model(model)
#         break

