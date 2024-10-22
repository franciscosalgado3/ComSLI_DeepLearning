#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#others
import numpy as np
import os
import matplotlib.pyplot as plt      # delete later after taking out the final plot 
from collections import OrderedDict

from config import paths
from loggs import logger
from visualization import plot_regression_test, compare_patches, compare_test_images, plot_all_test_metrics
from models.model_lstm_cnn import LSTM_CNN_2D, LSTM_CNN_3D
#from models.hybrid_lstm_cnn import HYBRID_LSTM_CNN_FLAT, HYBRID_LSTM_CNN_SLICE

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'       #set the visiible gpus in the hpc29 server used for training  


class Tester:
    def __init__(self, model: nn.Module, test_data: DataLoader, loss_function, device, plot_path, model_name, patch_size):
        """
        Initialize the Tester class.
        Args:
            model (nn.Module): The model to be trained.
            test_data (DataLoader): DataLoader for the testing dataset.
            loss_function: Loss function used to calculate the training/validation loss.
            device: Device on which to run the model (e.g., 'cuda' (gpus) or 'cpu').
            plot_path (str): Path to display training and validation loss plots.
            model_name (str): The name to use when saving the model.
            patch_size: Height/width of the tetsing patches.
        """
        self.logger = logger
        self.device = device
        self.model = model.to(self.device)
        self.test_data = test_data
        self.loss_function = loss_function
        self.plot_path = plot_path
        self.patch_size = patch_size
        self.model_name = model_name
    
    def evaluate(self):
        """
        Evaluate the model on the test dataset and calculate the average loss and R^2 score.
        Returns:
            tuple: The average loss and R^2 score across the entire test dataset.
        """
        self.model.eval()                   # Set the model to evaluation mode, disabling dropout and other training-specific behaviors
        run_loss, r2_total = 0.0, 0.0                # Initialize running totals for loss and R^2 score
        mask_targets, mask_outputs, diff_list, all_targets, all_outputs, all_profiles, all_metadata = [], [], [], [], [], [], []  # Initialize lists to store results

        with torch.no_grad():                                     # Disable gradient calculation for the evaluation loop (saves memory and computations)
            for source, target, metadata in self.test_data:
                source, target = source.to(self.device), target.to(self.device)          # Move data to the appropriate device 
                output, line_profiles = self.model(source)                          # Perform a forward pass through the model
                output = output % 180
                masked_output, masked_target = self._mask(output, target)                # Mask invalid values (-1) in the output and target
                diff = self._diff(masked_output.view(-1), masked_target.view(-1))
                loss = self.loss_function(diff, torch.zeros_like(diff))       # Compute the loss for the current batch. .view(-1) flattens the tensor in onde dimension
                run_loss += loss.item()                  # Accumulate the loss
                r2 = self._angular_r2_score(masked_target.view(-1), diff)         # Calculate the R^2 score for the current batch
                r2_total += r2                       # Accumulate the R^2 score
                
                # Store the results for later analysis and plotting
                mask_targets.extend(masked_target.cpu())      # .view(-1) flattens the tensor in onde dimension
                mask_outputs.extend(masked_output.cpu())
                diff_list.extend(diff.cpu())
                all_targets.extend(target.cpu().tolist())
                all_outputs.extend(output.cpu().tolist())
                all_profiles.extend(line_profiles.cpu().tolist())
                all_metadata.extend(metadata)
                            
        # Calculate the average loss and R^2 score across the entire test dataset
        average_loss = run_loss / len(self.test_data)
        average_r2 = r2_total / len(self.test_data)
        logger.info(f"Test Loss: {average_loss:.4f}, R^2: {average_r2:.4f}")
        
        # Plot regression results comparing masked targets and outputs
        # .cat concatenates all tensors, .cpu() moves tensor to cpu (for visualization), detach() takes tensor gradients out, and .numpy() transforms tensor into numpy array
        logger.info(f"Plotting regression results...")
        plot_regression_test(np.array(mask_targets), np.array(mask_outputs), np.array(diff_list), self.plot_path)
        
        logger.info(f"Plotting FOM comparison...")
        # Compare test images with model outputs and visualize the results
        compare_test_images(all_targets, all_outputs, all_metadata, self.plot_path, self.patch_size, self.logger)
        
        return average_loss, average_r2         # Return the average loss and R^2 score for further analysis or logging
    

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
    
def load_test_objects(dataset) -> tuple:
    """
    Load the necessary objects for testing, including the model, loss function, and paths for saving results.
    Args:
        dataset (str): The specific dataset or model variant to load for testing.
    Returns:
        tuple: A tuple containing the model, loss function, plot path, test path, and model name.
    """
    # Define the loss function to be used during testing
    loss_function = nn.SmoothL1Loss()

    # Uncomment the appropriate model instantiation based on the model architecture you want to load.
    #model = HYBRID_LSTM_CNN_FLAT(lstm_input_size=256, lstm_hidden_size=512, lstm_num_layers=4, channels=1, kernel_size=3, num_filters=8, dropout=0.1)
    #model = HYBRID_LSTM_CNN_SLICE(lstm_input_size=64, lstm_hidden_size=128, lstm_num_layers=4, channels=1, kernel_size=3, num_filters=8, dropout=0.1)
    model = LSTM_CNN_3D(lstm_input_size=1, lstm_hidden_size=8, lstm_num_layers=1, input_dim=1, hidden_dim=16, output_dim=16)
    #model = LSTM_CNN_2D(lstm_input_size=1, lstm_hidden_size=8, lstm_num_layers=1, channels=1, kernel_size=3, num_filters=8, dropout=0.1)
    
    # Get the model's name for saving and logging purposes
    model_name = model.get_name()
    
    # Define the main path where results will be stored, specific to the model
    main_path = os.path.join(paths['results'], model_name)
    
    # Define the path where the model checkpoints are stored for the specified dataset
    model_path = os.path.join(main_path, 'checkpoints', dataset)
    
    # Define the path where test results will be saved
    test_path = os.path.join(main_path, 'plots', 'testing')
    
    # Define the path for saving plots specific to the current dataset being tested
    plot_path = os.path.join(test_path, dataset)
    os.makedirs(plot_path, exist_ok=True)           # Create the directory if it doesn't exist
    
    # List all files in the model directory that have the .pt or .pth extension (model checkpoint files)
    files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.pt') or f.endswith('.pth')]
    if not files:              # Check if there are any model files in the directory
        raise FileNotFoundError("No model files found in the directory.")

    model_to_load = max(files, key=os.path.getmtime)       # Find the file with the most recent modification time (assuming it's the best/last checkpoint)
    state_dict = torch.load(model_to_load)             # Load the state dictionary from the checkpoint file
  
    # Create a new state dictionary without the 'module.' prefix, which is added by DataParallel. If data Paralell is not used this process can be excluded
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove the 'module.' prefix if it exists
        new_state_dict[name] = v
        
    # Load the state_dict into the model
    model.load_state_dict(new_state_dict)
      
    return model, loss_function, plot_path, test_path, model_name         # Return the model, loss function, plot path, test path, and model name as a tuple


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
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)

def load_dataset(path):
    """Load a dataset from a given path."""
    with open(path, 'rb') as f:
        data = torch.load(f)
    return data
    
def main():
    """
    Main function for evaluating the model on the test dataset across different cross-validation configurations.
    """
    
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()  # Clear the CUDA cache to free up memory
    
    # Check if multiple GPUs are available for data parallelism
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()  # Get the number of available GPUs
        logger.info(f"Using {num_gpus} GPUs for testing.")  # Log the number of GPUs being used

    # Load the test dataset
    logger.info("Loading testing set...")
    plot_dir =  os.path.join(paths['preprocessing'], 'datasets')
    
    test_set_path = os.path.join(plot_dir, 'test_set.pth')
    test_set = load_dataset(test_set_path)['testing']      # Load the testing set from the specified path
    
    val_set_path = os.path.join(plot_dir, 'val_set.pth')  # Path to the validation set
    val_set = load_dataset(val_set_path)['validation']
    
    final_test_set = test_set + val_set    # Load the testing set from the specified pat    
    test_data = create_dataloader(final_test_set, batch_size=num_gpus*(2**6), shuffle=False)  # Create a DataLoader for the test set

    # Define the different dataset configurations for testing (cross-validation setups)
    datasets = ['Augmented', 'Augmented + Contrast Adjust', 'Augmented + Gaussian Noise', 'Augmented + Speckle Noise', 'Augmented + Custom Noise']

    # Initialize lists to store the losses and R^2 scores for each dataset
    all_losses = []
    all_r2 = []
    
    logger.info("Testing with all cross-validation sets ..")

    # Loop through each dataset configuration for evaluation
    for i in range(len(datasets)):
        # Load the model, loss function, and paths for the current dataset configuration
        model, loss_function, plot_path, test_path, model_name = load_test_objects(datasets[i])
        model = nn.DataParallel(model)  # Enable data parallelism if multiple GPUs are available
        
        logger.info(f"Testing with {datasets[i]} dataset parameters on {model_name} model...")
        
        # Initialize the Tester class with the model and test data
        tester = Tester(model, test_data, loss_function, device, plot_path, model_name, patch_size=16)
        
        # Evaluate the model and store the loss and R^2 score
        loss, r2 = tester.evaluate()
        all_losses.append(loss)
        all_r2.append(r2)
        
    logger.info("All datasets were trained. PLotting final metrics..")  
    
    plot_all_test_metrics(all_losses, all_r2, datasets, test_path)
    
    logger.info("Testing pipeline finished.") 

if __name__=='__main__':
    main()