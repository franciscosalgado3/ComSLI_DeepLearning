#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

#others
import os
from config import paths
from loggs import logger
from early_stop import Early_Stopping
from visualization import plot_regression_results, plot_gan_losses, plot_patches
from models.lstm_gan import Generator, Discriminator

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"           #debugging
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'       #set the visiible gpus in the hpc29 server used for training 
 
class GAN_Trainer:
    def __init__(self, train_data: DataLoader, val_data: DataLoader, generator: nn.Module, discriminator: nn.Module, gen_optimizer: optim.Optimizer, gen_scheduler, disc_optimizer: optim.Optimizer, disc_scheduler, gen_loss, disc_loss, device, save_model_path, plot_path, gan_name, early_stopping, save_checkpoint, mask):
        """
        Initialize the GANTrainer class.
        """
        self.logger = logger
        self.device = device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.gen_train_losses = []
        self.gen_val_losses = []
        self.disc_train_real_losses = []
        self.disc_train_fake_losses = []
        self.adv_train_losses = []
        self.adv_val_losses = []

        self.save_model_path = save_model_path
        self.plot_path = plot_path
        self.gan_name = gan_name
        self.gen_name = 'Generator'
        self.disc_name = 'Discriminator'
        self.early_stopping = early_stopping
        self.save_checkpoint = save_checkpoint
        self.mask = mask
        
    def train(self, total_epochs: int):
        train_data = {}
        val_data = {}
        
        for epoch in range(total_epochs):
            disc_train_real_loss, disc_train_fake_loss, gen_train_loss, adv_train_loss, train_sources, train_targets, train_gan_outputs, train_metadata = self._run_epoch(self.train_data, epoch, train=True)
            gen_val_loss, adv_val_loss, val_sources, val_targets, val_gan_outputs, val_metadata = self._run_epoch(self.val_data, epoch, train=False)

            self.logger.info(f"- Epoch {epoch+1}/{total_epochs}")
            self.logger.info(f"  Training Loss  |  Generator (Huber Loss) {gen_train_loss:.4f}, Adversarial (BCE) {adv_train_loss:.4f}, Discriminator Real (BCE) {disc_train_real_loss:.4f}, Discriminator Fake (BCE) {disc_train_fake_loss:.4f}")
            self.logger.info(f" Validation Loss |  Generator (Huber Loss) {gen_val_loss:.4f}, Adversarial (BCE) {adv_val_loss:.4f}")

            self.gen_train_losses.append(gen_train_loss)
            self.adv_train_losses.append(adv_train_loss)
            self.disc_train_real_losses.append(disc_train_real_loss)
            self.disc_train_fake_losses.append(disc_train_fake_loss)
            self.gen_val_losses.append(gen_val_loss)
            self.adv_val_losses.append(adv_val_loss)
                        
            plot_gan_losses(self.gen_train_losses, self.adv_train_losses, self.gen_val_losses, self.adv_val_losses, self.disc_train_real_losses, self.disc_train_fake_losses, epoch, self.plot_path) 

            #self.gen_scheduler.step()
            #self.disc_scheduler.step()
            self.early_stopping(gen_val_loss, epoch + 1)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break
            
        # Save the best generator and final results
        self.save_checkpoint(self.generator, self.early_stopping.best_epoch, self.save_model_path, self.gen_name)
        self.early_stopping.load_best_model(self.generator, self.save_model_path, self.gen_name)
        # Save the best discriminator and final results
        self.save_checkpoint(self.discriminator, self.early_stopping.best_epoch, self.save_model_path, self.disc_name)
        self.early_stopping.load_best_model(self.discriminator, self.save_model_path, self.disc_name)
        
        train_data = {'source': train_sources, 'targets': train_targets, 'gan_outputs': train_gan_outputs, 'metadata': train_metadata}
        val_data = {'source': val_sources, 'targets': val_targets, 'gan_outputs': val_gan_outputs, 'metadata': val_metadata}
            
        # Save LSTM train and validation data as a dictionary
        torch.save(train_data, os.path.join(self.save_model_path, 'gan_train_data.pth'))
        torch.save(val_data, os.path.join(self.save_model_path, 'gan_val_data.pth'))
        return train_data, val_data    

    def _run_epoch(self, data_loader, epoch, train):
        disc_total_real_loss, disc_total_fake_loss, gen_total_loss, adv_total_loss = 0.0, 0.0, 0.0, 0.0
        num_batches = len(data_loader)
        all_sources, all_targets, all_gan_outputs, all_metadata = [], [], [], [], []
        # Number of discriminator updates per generator update
        discriminator_steps = 1 # adjust this value to control the frequency of discriminator updates
        
        with torch.set_grad_enabled(train):
            for batch_idx, (source, target, metadata) in enumerate(data_loader):
                source, target, = source.to(self.device), target.to(self.device),
                mask = self.mask(target).to(self.device)    # here binary value 1 for labeled pixels, 0 indicates missing pixels
                hint_vector = self._hint(mask).to(self.device)    # Create hint vector dynamically based on the mask: 1 for revelaed hints, 0 elsewhere
                all_metadata.extend(metadata)
                
                # Calculate the mean of observed angles for discriminator purposes
                real_angles = target * mask   # considers the evalated pixels only
                mean_angle = real_angles.mean() if real_angles.numel() > 0 else 90  # Default to 90 if no observed angles
                # Replace missing labels with the mean angle
                target_mean= target.clone()
                target_mean = torch.where(target == -1, mean_angle, target_mean)
                target_mean.to(self.device)
                
                if train:
                    ### Train Generator
                    self.generator.train()
                    self.gen_optimizer.zero_grad()        
                    
                    # Generate GAN output
                    gan_output = self.generator(source, target_mean, hint_vector)
                    gan_output = gan_output % 180
                    
                    # Train discriminator multiple times before updating the generator
                    for _ in range(discriminator_steps):    
                        self.discriminator.train()
                        self.disc_optimizer.zero_grad()   ### Train discriminator
                        
                        # Prepare real (soft) labels per pixel
                        real_labels = torch.full_like(target_mean, 0.9).to(self.device)  # Soft label for real data
                        # compute label pixels discriminator loss
                        real_output = self.discriminator(target_mean, hint_vector, mask)
                        pixel_disc_real_loss = self.disc_loss(real_output, real_labels)           # Discriminator loss on real data
                        disc_real_loss = (pixel_disc_real_loss * mask).sum() / mask.sum()
                        disc_total_real_loss = disc_total_real_loss + disc_real_loss.item()
                        
                        # Prepare fake (soft) labels per pixel
                        fake_labels = (mask * 0.9) + ((1 - mask) * 0.1)
                        # compute non labeled pixels discriminator loss
                        fake_output = self.discriminator(gan_output, hint_vector, mask)
                        pixel_disc_fake_loss = self.disc_loss(fake_output, fake_labels)           # Discriminator loss on fake data
                        disc_fake_loss = pixel_disc_fake_loss.mean()
                        disc_total_fake_loss = disc_total_fake_loss + disc_fake_loss.item()
                        
                        # determine final discriminator loss
                        disc_loss = disc_real_loss + disc_fake_loss
                        disc_loss.backward(retain_graph=True)
                        self.disc_optimizer.step()
                    
                    # Generator loss (adversarial loss + L2 loss)
                    # computing again these tensors to avoid errors in autograd mechanism when computing gradients
                    fake_output_2 = self.discriminator(gan_output, hint_vector, mask)
                    real_labels_2 = torch.full_like(target_mean, 0.9).to(self.device)  # Soft label for real data

                    # Generator loss (adversarial loss + L2 loss)
                    pixel_adv_disc_loss = self.disc_loss(fake_output_2, real_labels_2)
                    adv_fake_loss = pixel_adv_disc_loss.mean()
                    adv_total_loss = adv_total_loss + adv_fake_loss.item()
                    
                    diff = self._diff((gan_output * mask).view(-1), (target * mask).view(-1))
                    gen_loss = self.gen_loss(diff, torch.zeros_like(diff))   #masked
                    gen_total_loss = gen_total_loss + gen_loss.item()
                    
                    lambda_gen = 0.2 #higher (lower) value emphasizes reconstruction (adversarial) loss
                    gan_loss = adv_fake_loss * (1 - lambda_gen) + gen_loss * lambda_gen
                    gan_loss.backward()
                    self.gen_optimizer.step()
                                        
                    all_sources.append(source.cpu().detach())
                    all_targets.append(target.cpu().detach())
                    all_gan_outputs.append(gan_output.cpu().detach())
                    
                    if batch_idx == len(data_loader) - 1:
                            plot_patches(gan_output, target, epoch, self.plot_path, self.gan_name, train)
                            plot_regression_results(target[mask].view(-1).cpu().detach().numpy(), gan_output[mask].view(-1).cpu().detach().numpy(), diff.cpu().detach().numpy(), epoch, self.plot_path, train)
                        
                else:
                    self.generator.eval()
                    self.discriminator.eval()                   
                    with torch.no_grad():                        
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
                       
                        if batch_idx == len(data_loader) - 1:
                            plot_patches(gan_output, target, epoch, self.plot_path, self.gan_name, train)
                            #plot_regression_results(masked_target.view(-1).cpu().detach().numpy(), masked_output.view(-1).cpu().detach().numpy(), diff.cpu().detach().numpy(), epoch, self.plot_path, train)

                            
            all_sources = torch.cat(all_sources, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_gan_outputs = torch.cat(all_gan_outputs, dim=0)
            
        if train:
            return disc_total_real_loss/num_batches, disc_total_fake_loss/num_batches, gen_total_loss/num_batches, adv_total_loss/num_batches, all_sources, all_targets, all_gan_outputs, all_metadata
        else:
            return gen_total_loss/num_batches, adv_total_loss/num_batches, all_sources, all_targets, all_gan_outputs, all_metadata

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

def gan_mask(target):
    # Bynary mask for -1 values: are assigned with 0
    return (target != -1).float()

def angular_r2_score(target, diff):
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

def save_checkpoint(model, epoch, save_model_path, model_name):
    '''Save the model state to a checkpoint file.'''
    checkpoint_file = os.path.join(save_model_path, f"{model_name}_best_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_file)
                  
def load_train_objects() -> tuple:
    """
    Loads the model, optimizer, loss function, and model name for training.
    Returns:
        tuple: A tuple containing the model, optimizer, loss function, and model name.
    """
    ### GAN
    generator = Generator(input_channels=3, hidden_dim=16, output_dim=1, dropout=0.2)
    discriminator = Discriminator(input_channels=3, hidden_dim=16, output_dim=1, dropout=0.1)
    gen_optimizer = optim.AdamW(generator.parameters(), lr=5e-4)
    gen_scheduler = MultiStepLR(gen_optimizer, milestones=[50, 100], gamma=0.5)
    disc_optimizer = optim.AdamW(discriminator.parameters(), lr=1e-4)
    disc_scheduler = MultiStepLR(gen_optimizer, milestones=[50], gamma=0.75)
    gen_loss = nn.SmoothL1Loss(reduction='mean')   #or MSELoss()
    disc_loss = nn.BCELoss(reduction='none')
    gan_model_name = generator.get_name()
        
    return generator, discriminator, gen_optimizer, gen_scheduler, disc_optimizer, disc_scheduler, gen_loss, disc_loss, gan_model_name

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn,  drop_last=True)

def load_dataset(path):
    """
    Load a dataset from a given file path.
    Args:
        path (str): The file path to load the dataset from.
    Returns:
        Dataset: The loaded dataset.
    """
    with open(path, 'rb') as f:      # Open the file
        data = torch.load(f, weights_only=True)        # Load the dataset using torch's load function
    return data                  

def main():
    """
    Main function for training the model across different cross-validation datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()            # Clear the CUDA cache to free up memory
    if torch.cuda.device_count() >= 1:    
        num_gpus = torch.cuda.device_count()        # Get the number of available GPUs if more than 1
        logger.info(f"Using {num_gpus} GPUs for training and validation pipeline.") 
    
    # Set up directories for saving plots and models
    plot_dir =  os.path.join(paths['preprocessing'], 'datasets')
    datasets = ['Augmented'] #, 'Augmented + Contrast Adjust', 'Augmented + Gaussian Noise', 'Augmented + Speckle Noise', 'Augmented + Custom Noise']  # training is only done in augmented subset (baseline)
    
    # Create paths for each cross-validation fold's training set
    cv_folds_path = [os.path.join(plot_dir, f'train_set_{i}.pth') for i in range(len(datasets))]
    val_set_path = os.path.join(plot_dir, 'val_set.pth')  # Path to the validation set
    
    # Set up the main path for saving results
    results_path = os.path.join(paths['results'], 'training')
    os.makedirs(results_path, exist_ok=True)  # Create the directory if it doesn't exist
   
    # Load each cross-validation fold's training set into individual variables
    logger.info("Loading training fold sets...")
    train_dataloaders = [prepare_dataloader(load_dataset(fold_path), batch_size=num_gpus*(2**7), shuffle=True) for fold_path in cv_folds_path]
    # Load the validation set
    logger.info("Loading validation set...")
    val_set = load_dataset(val_set_path)['validation']
    val_data = prepare_dataloader(val_set, batch_size=num_gpus*(2**7), shuffle=False)
    
    logger.info("Loading model objects..")
    generator, discriminator, gen_optimizer, gen_scheduler, disc_optimizer, disc_scheduler, gen_loss, disc_loss, gan_name = load_train_objects()
    
    # Initialize early stopping. Adjust parameters as needed.
    early_stopping_2 = Early_Stopping(patience=20, verbose=False, delta=0.0001)
    
    logger.info(f"Iniciating training and validating for GAN training - {datasets[0]}...")
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)    
    
    # Set up paths for saving model checkpoints and plots
    main_path = os.path.join(results_path, gan_name)
    save_gan_path = os.path.join(main_path, 'checkpoints')
    os.makedirs(save_gan_path, exist_ok=True)
    gan_plot_path = os.path.join(main_path, 'plots')
    os.makedirs(gan_plot_path, exist_ok=True)
    # Set up paths specific to the current fold
    save_fold_gan_path = os.path.join(save_gan_path, f'{datasets[0]}')
    os.makedirs(save_fold_gan_path, exist_ok=True)
    plot_gan_path = os.path.join(gan_plot_path, f'{datasets[0]}')
    os.makedirs(plot_gan_path, exist_ok=True)
    
    gan_trainer = GAN_Trainer(train_dataloaders[0], val_data, generator, discriminator, gen_optimizer, gen_scheduler, disc_optimizer, disc_scheduler, gen_loss, disc_loss, device, save_fold_gan_path, plot_gan_path, gan_name, early_stopping_2, save_checkpoint, gan_mask)
    gan_trainer.train(total_epochs=400)
    
    logger.info("Training and validation pipeline finished.") 
    
if __name__ == '__main__':
    main()