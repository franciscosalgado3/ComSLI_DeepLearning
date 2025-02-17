#torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#others
import os
import numpy as np
from config import paths
from loggs import logger
from visualization import plot_regression_test, compare_test_images
from models.lstm_gan import Generator, Discriminator 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class GAN_Tester:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, test_data: DataLoader, gen_loss, disc_loss, gan_name, device, plot_path, test_dir, patch_size):
        self.logger = logger
        self.device = device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.test_data = test_data
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.plot_path = plot_path
        self.gan_name = gan_name
        self.patch_size = patch_size
        self.test_dir = test_dir
        
    def evaluate(self):
        all_sources, all_targets, all_metadata, all_gan_outputs = [], [], [], []
        gen_total_loss, adv_total_loss = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (source, target, metadata) in enumerate(self.test_data):
                source, target = source.to(self.device), target.to(self.device)
                mask = self._mask(target).to(self.device)
                hint_vector = self._hint(mask).to(self.device)    # Create hint vector dynamically based on the mask
                all_metadata.extend(metadata)
                
                # Calculate the mean of observed angles for discriminator purposes
                real_angles = target * (1 -mask)
                mean_angle = real_angles.mean() if real_angles.numel() > 0 else 90  # Default to 90 if no observed angles
                # Replace missing labels with the mean angle
                target_mean= target.clone()
                target_mean = torch.where(target == -1, mean_angle, target_mean)
                target_mean.to(self.device)
                       
                # Generate GAN output
                gan_output = self.generator(source, target_mean, hint_vector)
                gan_output = gan_output % 180
                
                # compute reconstruction loss on labeled pixels
                diff = self._diff((gan_output * mask).view(-1), (target * mask).view(-1))
                gen_loss = self.gen_loss(diff, torch.zeros_like(diff))   #masked
                gen_total_loss += gen_loss.item()
                
                # adversarial discriminator loss on hard label 
                fake_output = self.discriminator(gan_output, hint_vector, mask)
                adv_loss = self.disc_loss(fake_output, torch.ones_like(fake_output))
                adversarial_loss = adv_loss.mean()
                adv_total_loss += adversarial_loss.item()                        

                all_sources.append(source.cpu().detach())
                all_targets.append(target.cpu().detach())
                all_gan_outputs.append(gan_output.cpu().detach())
                
        average_gen_loss = gen_total_loss / len(self.test_data)
        average_adv_loss = adv_total_loss / len(self.test_data)
        self.logger.info(f"Average Generator Loss: {average_gen_loss:.4f}")
        self.logger.info(f"Average Adversarial Loss: {average_adv_loss:.4f}")
        
        logger.info(f"Plotting FOM comparison...")
        # Compare test images with model outputs and visualize the results
        compare_test_images(all_targets, all_gan_outputs, all_metadata, self.plot_path, self.patch_size, self.logger)
        plot_regression_test(target[mask].view(-1).cpu().detach().numpy(), gan_output[mask].view(-1).cpu().detach().numpy(), diff.cpu().detach().numpy(), self.plot_path)

        #torch for dataset creation for gan testing
        test_gan_outputs = torch.tensor(all_gan_outputs)
        test_targets = torch.tensor(all_targets)
        
        test_data = {'targets': test_targets, 'gan_outputs': test_gan_outputs, 'metadata': all_metadata}
        # Save LSTM train and validation data as a dictionary
        torch.save(test_data, os.path.join(self.test_dir, 'lstm_test_data.pth'))

        
    def _gan_mask(target):
        # Bynary mask for -1 values: are assigned with 0
        return (target != -1).float()

    def _angular_r2_score(target, diff):
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
    
    def _hint(self, mask, min_hint_fraction=0.0, base_hint_fraction=0.1, max_hint_fraction=1.0):
        """
        Create a dynamic hint mask based on the number of evaluated pixels in each 16x16 patch.
        Args:
            mask (Tensor): The binary mask indicating evaluated pixels (1 for unlabeled, 0 for the evaluated).
                           Shape: [batch_size, 16, 16]
            min_hint_fraction (float, optional): Hint fraction when p_u ≤ 20. Defaults to 0.0.
            base_hint_fraction (float, optional): Base hint fraction for 20 < p_u ≤ 240. Defaults to 0.1.
            max_hint_fraction (float, optional): Hint fraction when p_u > 240. Defaults to 1.0.
        Returns:
            Tensor: The hint mask where some evaluated pixels are revealed based on H.
        """
        # Ensure mask is a float tensor
        mask = (1 - mask).float()

        # Calculate the number of evaluated pixels (p_u) for each patch
        p_u = mask.view(mask.size(0), -1).sum(dim=1)

        # Compute hint fraction H based on p_u using the piecewise function
        H = torch.where(
            p_u <= 20,
            torch.full_like(p_u, min_hint_fraction),
            torch.where(
                p_u <= 240,
                base_hint_fraction + ((p_u - 20) / 220.0) * (max_hint_fraction - base_hint_fraction),
                torch.full_like(p_u, max_hint_fraction)
            )
        )

        # Clamp H to ensure it stays within [0, 1]
        H = H.clamp(0.0, 1.0)

        # Reshape H to [batch_size, 1, 1] for broadcasting
        H = H.view(-1, 1, 1)

        # Generate a random tensor for comparison, same shape as mask
        rand = torch.rand(mask.size(), device=self.device)

        # Create hint mask: reveal pixels where rand < H and mask == 1
        hint_mask = (rand < H) & (mask == 1)

        # Generate the final hint vector by masking
        hint_vector = mask * hint_mask.float()

        return hint_vector  

    
def load_gan_objects(train_dir, test_gan_path, dataset) -> tuple:
    generator = Generator(input_channels=3, hidden_dim=16, output_dim=1, dropout=0.2)
    discriminator = Discriminator(input_channels=3, hidden_dim=16, output_dim=1, dropout=0.2)
    gen_loss = nn.SmoothL1Loss(reduction='mean')
    disc_loss = nn.BCELoss(reduction='none')
    gan_model_name = generator.get_name()
    gen_name = 'Generator'
    disc_name = 'Discriminator'
    
    # Define the path where the model checkpoints are stored for the specified dataset
    model_gan_path = os.path.join(train_dir, gan_model_name, 'checkpoints', dataset)
    
    plot_gan_path = os.path.join(test_gan_path, dataset)
    os.makedirs(plot_gan_path, exist_ok=True)
    
    # List all model files in the directory, filtering for .pt or .pth
    files = [os.path.join(model_gan_path, f) for f in os.listdir(model_gan_path) if f.endswith('.pt') or f.endswith('.pth')]
    # Filter files for generator and discriminator
    gen_files = [f for f in files if os.path.basename(f).startswith(gen_name)]
    disc_files = [f for f in files if os.path.basename(f).startswith(disc_name)]
    
    if not gen_files:
        raise FileNotFoundError(f"No generator files found starting with '{gen_name}'")
    if not disc_files:
        raise FileNotFoundError(f"No discriminator files found starting with '{disc_name}'")
    
    # Get the most recent generator and discriminator files based on modification time
    gen_model_to_load = max(gen_files, key=os.path.getmtime)
    disc_model_to_load = max(disc_files, key=os.path.getmtime)
    
    # Load the state dicts for both generator and discriminator
    gen_state_dict = torch.load(gen_model_to_load)
    disc_state_dict = torch.load(disc_model_to_load)
    
    # Remove 'module.' from the state_dict keys if the model was saved with DataParallel
    gen_new_state_dict = {key.replace("module.", ""): value for key, value in gen_state_dict.items()}
    disc_new_state_dict = {key.replace("module.", ""): value for key, value in disc_state_dict.items()}
    
    # Load the state dicts into the models
    generator.load_state_dict(gen_new_state_dict)
    discriminator.load_state_dict(disc_new_state_dict)
    
    return generator, discriminator, gen_loss, disc_loss, gan_model_name, plot_gan_path

def custom_collate(batch):
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

def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle=True)-> DataLoader:
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)

def load_dataset(path):
    """Load a dataset from a given path."""
    with open(path, 'rb') as f:
        data = torch.load(f, weights_only=True)
    return data

def main():
    """
    Main function for evaluating the model on the test dataset across different cross-validation configurations and logging the results to a CSV file.
    """
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()  # Clear the CUDA cache to free up memory
    
    # Check if multiple GPUs are available for data parallelism
    if torch.cuda.device_count() >= 1:
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
    test_data = prepare_dataloader(final_test_set, batch_size=num_gpus*(2**7), shuffle=False)  # Create a DataLoader for the test set
    
    # training and testing dir
    train_dir = os.path.join(paths['results'], 'training')
    test_dir = os.path.join(paths['results'], 'testing')
    
    # Define the different dataset configurations for testing (cross-validation setups)
    datasets = ['Augmented'] #, 'Augmented + Contrast Adjust', 'Augmented + Gaussian Noise', 'Augmented + Speckle Noise', 'Augmented + Custom Noise']
        
    ###### GAN  ######## 
    
    logger.info("Testing GAN model on non-labeled pixels. Loading objects..")
    
    generator, discriminator, gen_loss, disc_loss, gan_model_name, plot_gan_path = load_gan_objects(train_dir, test_dir, datasets[0])
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    
    logger.info(f"Testing with {datasets[0]} dataset parameters using {gan_model_name} model...")
    
    # Initialize the Tester class with the model and test data
    tester = GAN_Tester(generator, discriminator, test_data, gen_loss, disc_loss, gan_model_name, device, plot_gan_path, test_dir, patch_size=16)
    tester.evaluate()

    logger.info("Training and validation pipeline finished.") 
    
if __name__ == '__main__':
    main()