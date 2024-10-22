#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#others
import os
import matplotlib.pyplot as plt      # delete later after taking out the final plot 

from config import paths
from loggs import logger
from early_stop import Early_Stopping
from visualization import plot_regression_results, plot_losses, plot_all_train_losses
from models.model_lstm_cnn import LSTM_CNN_2D, LSTM_CNN_3D
#from models.hybrid_lstm_cnn import HYBRID_LSTM_CNN_FLAT, HYBRID_LSTM_CNN_SLICE


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"           #debugging
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'       #set the visiible gpus in the hpc29 server used for training  

class Trainer:
    def __init__(self, model: nn.Module, train_data: DataLoader, val_data: DataLoader, optimizer: optim.Optimizer, loss_function, device, save_model_path, plot_path, early_stopping, model_name):
        """
        Initialize the Trainer class.
        Args:
            model (nn.Module): The model to be trained.
            train_data (DataLoader): DataLoader for the training dataset.
            val_data (DataLoader): DataLoader for the validation dataset.
            optimizer (optim.Optimizer): Optimizer used for model training.
            loss_function: Loss function used to calculate the training/validation loss.
            device: Device on which to run the model (e.g., 'cuda' (gpus) or 'cpu').
            save_model_path (str): Path to save the model checkpoints.
            plot_path (str): Path to display training and validation loss plots.
            early_stopping (Early_Stopping): Early_Stopping object to monitor training.
            model_name (str): The name to use when saving the model.
        """
        self.logger = logger
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.save_model_path = save_model_path
        self.train_losses = []
        self.val_losses = []
        self.plot_path = plot_path
        self.early_stopping = early_stopping
        self.model_name = model_name
        
    def train(self, total_epochs: int):
        """
        Train the model for a specified number of epochs.
        Args:
            total_epochs (int): Number of epochs to train the model.
        """
        for epoch in range(total_epochs):
            # Run training and validation for each epoch (see function _run_epoch)
            train_loss, train_r2 = self._run_epoch(self.train_data, epoch, train=True)
            val_loss, val_r2 = self._run_epoch(self.val_data, epoch, train=False)

            # Log training and validation results for the epoch
            logger.info(f"- Epoch {epoch+1}/{total_epochs}")
            logger.info(f" Training Loss: {train_loss:.4f}, R^2: {train_r2:.4f}")
            logger.info(f" Validation Loss: {val_loss:.4f}, R^2: {val_r2:.4f}")

            # Store losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Plot the training and validation losses
            plot_losses(self.train_losses, self.val_losses, epoch, self.plot_path)        
            
            # Check if early stopping should be triggered. Comment this line if dont want to use this functionality
            self.early_stopping(val_loss, epoch + 1)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break
            
        # Save the best model and final results
        self._save_checkpoint(self.model, self.early_stopping.best_epoch, self.save_model_path, self.model_name)
        self.early_stopping.load_best_model(self.model, self.save_model_path, self.model_name)
        
        return self.train_losses, self.val_losses  # Return the recorded losses for further analysis
  
    def _run_epoch(self, data_loader, epoch, train):
        """
        Run a single training or validation epoch.
        Args:
            data_loader (DataLoader): DataLoader for the current epoch's dataset.
            epoch (int): Current epoch number.
            train (bool): Whether this is a training epoch (True) or validation epoch (False).
        
        Returns:
            tuple: The average loss and R^2 score for the epoch.
        """
        train_loss, r2_train_total = 0.0, 0.0
        val_loss, r2_val_total = 0.0, 0.0
        num_batches = len(data_loader)       # Number of batches in the DataLoader

        with torch.set_grad_enabled(train):         # Enable/disable gradient calculation based on training mode (if train enables)
            for batch_idx, (source, target, _) in enumerate(data_loader):            # Iterate data in data_loader (not metadata because in trainin is not used)
                source, target = source.to(self.device), target.to(self.device)       # Move data to the specified device

                if train:
                    self.model.train()                  # Set the model to training mode
                    self.optimizer.zero_grad()           # Clear previous gradients
                    output, _ = self.model(source)              # Forward pass of image data through the model obtaining output prediction labels
                    output = output % 180                             # Ensure the predictions are within the valid range [0, 180)
                    masked_output, masked_target = self._mask(output, target)       # Mask invalid values (-1) for loss calculation and R^2 calculation
                    diff = self._diff(masked_output.view(-1), masked_target.view(-1))
                    loss = self.loss_function(diff, torch.zeros_like(diff))       # Compute the loss for the current batch. .view(-1) flattens the tensor in onde dimension
                    train_loss += loss.item()                # Accumulate the loss
                    r2_train = self._angular_r2_score(masked_target.view(-1), diff)  # Calculate R^2. .view(-1) flattens the tensor in onde dimension
                    r2_train_total += r2_train                  # Accumulate the R^2 score
                    loss.backward()                      # Backward pass (calculate gradients)
                    self.optimizer.step()                       # Update model weights                    
                    if batch_idx == len(data_loader) - 1:           # Plot the last batch's results for visualization (for convinience)
                        # .cpu() moves tensor to cpu (for visualization), detach() takes tensor gradients out, and .numpy() transforms tensor in numpy array
                        plot_regression_results(masked_target.view(-1).cpu().detach().numpy(), masked_output.view(-1).cpu().detach().numpy(), diff.cpu().detach().numpy(), epoch, self.plot_path, train)
                                
                else:
                    self.model.eval()                         # Set the model to evalu  ation mode
                    with torch.no_grad():                        # Disable gradient calculation
                        output, _ = self.model(source)                   # Forward pass through the model
                        output = output % 180                             # Ensure the predictions are within the valid range [0, 180)
                        masked_output, masked_target = self._mask(output, target)        # Mask invalid values (-1) for loss calculation and R^2 calculation
                        diff = self._diff(masked_output.view(-1), masked_target.view(-1))
                        loss = self.loss_function(diff, torch.zeros_like(diff))       # Compute the loss for the current batch. .view(-1) flattens the tensor in onde dimension
                        val_loss += loss.item()          # Accumulate the loss
                        r2_val = self._angular_r2_score(masked_target.view(-1), diff)  # Calculate R^2
                        r2_val_total += r2_val              # Accumulate the R^2 score
                        if batch_idx == len(data_loader) - 1:           # Plot the last batch's results for visualization
                            plot_regression_results(masked_target.view(-1).cpu().numpy(), masked_output.view(-1).cpu().numpy(), diff.cpu().numpy(), epoch, self.plot_path, train)     
       
        if train:
            return train_loss / num_batches, r2_train_total / num_batches           # Return average training loss and R^2 score
        else:
            return val_loss / num_batches, r2_val_total / num_batches               # Return average validation loss and R^2 score
                   
    def _diff(self, output, target):
        """
        Calculate the angular difference between two angles in degrees.
        Args:
            output (Tensor): Model output predictions (angles in degrees).
            target (Tensor): Ground truth target values (angles in degrees).
        Returns:
            float: The angular difference in degrees.
        """
        # Compute the circular difference
        diff = torch.abs(output - target)
        diff = torch.min(diff, 180 - diff)  # Handle cyclic nature
        return diff
        
    
    def _angular_r2_score(self, target, diff):
        """
        Calculate the R^2 (coefficient of determination) score for circular data in the range [0, 180).
        Args:
            output (Tensor): Model output predictions (angles in degrees).
            target (Tensor): Ground truth target values (angles in degrees).
        Returns:
            float: The R^2 score.
        """
        # Compute the mean of the target values
        target_mean = torch.mean(target)
        # Total sum of squares
        ss_tot = torch.sum((target - target_mean) ** 2)
        # Residual sum of squares
        ss_res = torch.sum(diff ** 2)
        # R^2 score
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared.item()    

    
    def _calculate_r2(self, output, target):
        """
        Calculate the R^2 (coefficient of determination) score.
        Args:
            output (Tensor): Model output predictions.
            target (Tensor): Ground truth target values.
        Returns:
            float: The R^2 score.
        """
        target_mean = torch.mean(target)                     # Calculate the mean of the target values
        ss_tot = torch.sum((target - target_mean) ** 2)      # Calculate total sum of squares
        ss_res = torch.sum((target - output) ** 2)           # Calculate residual sum of squares
        r_squared = 1 - ss_res / ss_tot                      # Calculate R^2 score
        return r_squared.item()                           # Return the R^2 score as a float
    

    def _save_checkpoint(self, model, epoch, save_model_path, model_name) -> None:        
        """
        Save the model state to a checkpoint file.
        Args:
            model (nn.Module): The model instance to save.
            epoch (int): The epoch number associated with the saved model.
            save_model_path (str): Directory path to save the model checkpoint.
            model_name (str): The name to use for the saved model file.
        """
        checkpoint_file = os.path.join(save_model_path, f"{model_name}_best_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_file)  # Save the model's state dictionary
            
    def _mask(self, output, target):
        """
        Mask invalid values (-1) from the output and target tensors.
        Args:
            output (Tensor): Model output predictions.
            target (Tensor): Ground truth target values.
        Returns:
            tuple: Masked output and target tensors.
        """
        valid_mask = (target != -1)             # Create a mask for valid target values (where target is not -1)
        masked_output = output[valid_mask]       # Apply the mask to the output
        masked_target = target[valid_mask]       # Apply the mask to the target
        return masked_output, masked_target     # Return the masked output and target tensors
               

def load_train_objects() -> tuple:    
    """
    Loads the model, optimizer, loss function, and model name for training.
    Returns:
        tuple: A tuple containing the model, optimizer, loss function, and model name.
    """
    # Initialize the model to be used for training. Uncomment the appropriate model to use.
    # model = HYBRID_LSTM_CNN_FLAT(lstm_input_size=256, lstm_hidden_size=512, lstm_num_layers=4, channels=1, kernel_size=3, num_filters=8, dropout=0.1)        # this model gives worst results
    # model = HYBRID_LSTM_CNN_SLICE(lstm_input_size=64, lstm_hidden_size=128, lstm_num_layers=4, channels=1, kernel_size=3, num_filters=8, dropout=0.1)        # this model also gives worst results
    model = LSTM_CNN_3D(lstm_input_size=1, lstm_hidden_size=8, lstm_num_layers=1, input_dim=1, hidden_dim=16, output_dim=16)
    # model = LSTM_CNN_2D(lstm_input_size=1, lstm_hidden_size=8, lstm_num_layers=1, channels=1, kernel_size=3, num_filters=8, dropout=0.1)
    
    #  Adam optimizer is used here with a learning rate of 1e-4.
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Uncomment the appropriate loss function.
    # mse_loss = nn.MSELoss()       # Mean Squared Error Loss (MSE)
    # mae_loss = nn.L1Loss()        # Mean Absolute Error Loss (MAE)
    hub_loss = nn.SmoothL1Loss()    # Huber Loss (Smooth L1 Loss)

    model_name = model.get_name()        # Get the name of the model for saving and logging purposes

    return model, optimizer, hub_loss, model_name

def custom_collate_fn(batch):
    """
    Custom collate function to process and batch the data.
    Handle diferences in metadata if there is so (e.g. if not all patches have 'mean_angle_patch')
    Args:
        batch (list): A list of tuples (source, target, metadata) representing a batch of data.
    Returns:
        tuple: Collated sources, targets, and metadata, where sources and targets are batched tensors,
               and metadata is a list of dictionaries.
    """
    # Initialize lists for collated data
    collated_sources = []
    collated_targets = []
    collated_metadata = []

    # Loop over each item in the batch
    for item in batch:
        source, target, metadata = item     # Unpack the source, target, and metadata
        collated_sources.append(source)     # Append the source tensor to the collated sources list
        collated_targets.append(target)     # Append the target tensor to the collated targets list

        # Ensure 'mean_angle_patch' is present in metadata
        if 'mean_angle_patch' not in metadata:
            metadata['mean_angle_patch'] = None    # Add None if 'mean_angle_patch' is missing

        collated_metadata.append(metadata)  # Append the metadata dictionary to the collated metadata list

    # Use the default_collate function from PyTorch to batch the source and target tensors
    collated_sources = torch.utils.data.dataloader.default_collate(collated_sources)
    collated_targets = torch.utils.data.dataloader.default_collate(collated_targets)

    return collated_sources, collated_targets, collated_metadata
 
def create_dataloader(data, batch_size, shuffle=True):
    """
    Create a DataLoader for the given dataset with a custom collate function.
    Drop_last = True ensures last batch is not used for calculations (avoid errors in gpu parallelization if theres a discrepancy in number of samples regarding other batches)
    Args:
        data (Dataset): The dataset to be loaded.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch. Default is True.
    Returns:
        DataLoader: A DataLoader object for the dataset.
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn,  drop_last=True)

def load_dataset(path):
    """
    Load a dataset from a given file path.
    Args:
        path (str): The file path to load the dataset from.
    Returns:
        Dataset: The loaded dataset.
    """
    with open(path, 'rb') as f:      # Open the file
        data = torch.load(f)        # Load the dataset using torch's load function
    return data                  


def main():
    """
    Main function for training the model across different cross-validation datasets.
    """
    
    # Set the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()            # Clear the CUDA cache to free up memory
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()        # Get the number of available GPUs if more than 1
        logger.info(f"Using {num_gpus} GPUs for training and validation pipeline.") 

    # Set up directories for saving plots and models
    plot_dir =  os.path.join(paths['preprocessing'], 'datasets')
    datasets = ['Augmented', 'Augmented + Contrast Adjust', 'Augmented + Gaussian Noise', 'Augmented + Speckle Noise', 'Augmented + Custom Noise']
    
    # Create paths for each cross-validation fold's training set
    cv_folds_path = [os.path.join(plot_dir, f'train_set_{i}.pth') for i in range(len(datasets))]
    val_set_path = os.path.join(plot_dir, 'val_set.pth')  # Path to the validation set
    
    # Set up the main path for saving results
    results_path = paths['results']
    os.makedirs(results_path, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Load each cross-validation fold's training set into individual variables
    logger.info("Loading training fold sets...")
    # Adjust batch_size if memory problems (decrease by a power of 2 factor)
    train_dataloaders = [create_dataloader(load_dataset(fold_path), batch_size=num_gpus*(2**6), shuffle=True) for fold_path in cv_folds_path]

    # Load the validation set
    logger.info("Loading validation set...")
    val_set = load_dataset(val_set_path)['validation']
    # Adjust batch_size according to the trainloader batch_size
    val_data = create_dataloader(val_set, batch_size=num_gpus*(2**6), shuffle=False)

    # Initialize lists to store training and validation losses across all folds
    all_train_losses = []
    all_val_losses = []
    
    logger.info(f"Iniciating training and validating pipeline.")
    
    # Loop through each fold (cross-validation dataset) for training and validation
    for i, train_dataloader in enumerate(train_dataloaders):
        logger.info(f"Training on fold {i}: {datasets[i]}.")

        # Reinitialize model and optimizer for each fold
        logger.info("Loading model and objects..")
        model, optimizer, loss_function, model_name = load_train_objects()
        model = nn.DataParallel(model)     # Enable data parallelism for multi-GPU training
        
        # Set up paths for saving model checkpoints and plots
        main_path = os.path.join(results_path, model_name)
        save_model_path = os.path.join(main_path, 'checkpoints')
        os.makedirs(save_model_path, exist_ok=True)
        model_plot_path = os.path.join(main_path, 'plots')
        os.makedirs(model_plot_path, exist_ok=True)
        plot_path = os.path.join(model_plot_path, 'training')
        os.makedirs(plot_path, exist_ok=True)

        # Initialize early stopping. Adjust parameters as needed.
        early_stopping = Early_Stopping(patience=10, verbose=False, delta=0.0001)

        # Set up paths specific to the current fold
        save_fold_model_path = os.path.join(save_model_path, f'{datasets[i]}')
        os.makedirs(save_fold_model_path, exist_ok=True)
        plot_fold_path = os.path.join(plot_path, f'{datasets[i]}')
        os.makedirs(plot_fold_path, exist_ok=True)

        # Initialize the trainer and start training
        trainer = Trainer(model, train_dataloader, val_data, optimizer, loss_function, device, save_fold_model_path, plot_fold_path, early_stopping, model_name)
        # Adjust total_epoch parameters as wanted
        train_losses, val_losses = trainer.train(total_epochs=100)

        # Store the losses for this fold
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
      
    logger.info("All datasets were trained. PLotting all losses combined..")  
    
    plot_all_train_losses(all_train_losses, all_val_losses, datasets, plot_path)
    
    logger.info("Training and validation pipeline finished.") 
    
if __name__ == '__main__':
    main()