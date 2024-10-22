#matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, ListedColormap
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

# others
from sklearn.linear_model import LinearRegression
import os
import numpy as np
from PIL import Image
import imageio
import torch


#############     PRE-PROCESSING VISUALIZATION FUNCTIONS      #############

def plot_occurrences(occurrences, min_occurrences, threshold, file, plot_path):
    """
    This function visualizes the distribution of angle occurrences in the dataset. 
    It creates a bar chart showing the frequency of each patych mean angle, along with lines indicating the minimum occurrences 
    and a threshold for balancing the dataset. The plot is saved to the specified directory.
    """
    # Create a directory for the plots if it doesn't already exist
    plot_dir = os.path.join(plot_path, 'data_augmentation/')
    os.makedirs(plot_dir, exist_ok=True)
    
    angles = list(occurrences.keys())
    counts = list(occurrences.values())
    plt.bar(angles, counts, color='blue')
    plt.axhline(y=min_occurrences, color='green', linestyle='--', label='Min Occurrences')
    plt.axhline(y=threshold, color='red', linestyle='-', label='Threshold')
    plt.title(file)
    plt.xlabel('Mean Angle')
    plt.ylabel('Count')
    plt.legend(loc=2)
    plt.savefig(os.path.join(plot_dir, f'{file}.png'))
    plt.close()
        
def plot_raw_data(image, label, pair_name, plot_path, train_flag):
    """
    This function visualizes raw image data alongside its corresponding label. 
    It generates a subplot with the real image, the label image, and a scatter plot of the label values. 
    The resulting plot is saved to the appropriate directory based on whether the data is for training or testing.
    """
    plot_dir = os.path.join(plot_path, 'data/','raw_data/')
    os.makedirs(plot_dir, exist_ok=True)
    if train_flag:
        plot_dir = os.path.join(plot_dir, 'train/')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = os.path.join(plot_dir, 'test/')
        os.makedirs(plot_dir, exist_ok=True)   
    
    # Plot settings
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Plot the real image (first slice for simplicity)
    axs[0].imshow(image[:, :, 0], cmap='gray')
    axs[0].set_title('Real Image (1st Slice)')
    axs[0].axis('off')
    # Plot the label image (first slice for simplicity)
    axs[1].imshow(label[:, :], cmap='gray')
    axs[1].set_title('Label Image')
    axs[1].axis('off')
    # Scatter plot of the flattened label values
    axs[2].scatter(range(len(label.flatten())), label.flatten(), alpha=0.6, marker='.')
    axs[2].set_title('Scatter Plot of Angle Values')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Label Value')
    # Save the plot
    plt.savefig(os.path.join(plot_dir, f'{pair_name}.png'))
    plt.close(fig)
    
def plot_all_patches(tiff_data, vis_image, data, file, plot_path, train_flag):
    """
    This function visualizes all patches selected during preprocessing. 
    It generates a blended image showing the patches overlaid on the original TIFF data, 
    along with a scatter plot showing the distribution of angle values across raw, modified, and unmodified patches. 
    The plot is saved to the appropriate directory based on whether the data is for training or testing.
    """
    plot_dir = os.path.join(plot_path, 'data/', 'all_patches/')
    os.makedirs(plot_dir, exist_ok=True)
    if train_flag:
        plot_dir = os.path.join(plot_dir, 'train/')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = os.path.join(plot_dir, 'test/')
        os.makedirs(plot_dir, exist_ok=True)  
    
    plt.figure(figsize=(12, 6))
    # Plot blended image
    plt.subplot(1, 2, 1)
    tiff_rgb = np.stack((tiff_data,)*3, axis=-1)
    blended_image = (0.5 * vis_image + 0.5 * tiff_rgb).astype(np.uint8)
    plt.imshow(blended_image)
    plt.title('Selected Patches')
    plt.axis('off')
    # Plot scatter of angles for training and testing
    plt.subplot(1, 2, 2)
    raw_data = [data[1].flatten() for data in data if data[2]['nature'] == 'raw']
    mod_data = [data[1].flatten() for data in data if data[2]['nature'] == 'modified']
    nmod_data = [data[1].flatten() for data in data if data[2]['nature'] == 'not modified']
    if len(raw_data) > 1:
        raw_data = np.concatenate(raw_data)
        plt.scatter(range(len(raw_data)), raw_data, color='green', label='Raw', alpha=0.5)
    if len(mod_data) > 1:
        mod_data = np.concatenate(mod_data)
        plt.scatter(range(len(mod_data)), mod_data, color='blue', label='Modified', alpha=0.5)
    if len(nmod_data) > 1:
        nmod_data = np.concatenate(nmod_data)
        plt.scatter(range(len(nmod_data)), nmod_data, color='red', label='Not modified', alpha=0.5)
    plt.legend(loc=2)
    plt.title('Scatter Plot of Angles')
    plt.xlabel('Index')
    plt.ylabel('Angle')
    # Save the plots
    plt.savefig(os.path.join(plot_dir, f'{file}.png'))
    plt.close()
      
def plot_good_patches(tiff_data, vis_image, data, file, plot_path, train_flag):
    """
    This function visualizes the 'good' patches selected during preprocessing, including discarded ones. 
    It generates a blended image showing the selected patches, and a scatter plot that distinguishes between 
    raw, modified, not modified, and discarded patches based on their angle values. 
    The plot is saved to the appropriate directory based on whether the data is for training or testing.
    """
    plot_dir = os.path.join(plot_path, 'data/', 'good_patches/')
    os.makedirs(plot_dir, exist_ok=True)
    if train_flag:
        plot_dir = os.path.join(plot_dir, 'train/')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = os.path.join(plot_dir, 'test/')
        os.makedirs(plot_dir, exist_ok=True)  
    
    plt.figure(figsize=(12, 6))
    # Plot blended image
    plt.subplot(1, 2, 1)
    tiff_rgb = np.stack((tiff_data,)*3, axis=-1)
    blended_image = (0.5 * vis_image + 0.5 * tiff_rgb).astype(np.uint8)
    plt.imshow(blended_image)
    plt.title('Selected Patches')
    plt.axis('off')
    # Plot scatter of angles for training and testing
    plt.subplot(1, 2, 2)
    raw_data = [data[1].flatten() for data in data if data[2]['nature'] == 'raw']
    mod_data = [data[1].flatten() for data in data if data[2]['nature'] == 'modified']
    nmod_data = [data[1].flatten() for data in data if data[2]['nature'] == 'not modified']
    disc_data = [data[1].flatten() for data in data if data[2]['nature'] == 'discarded']
    if len(raw_data) > 1:
        raw_data = np.concatenate(raw_data)
        plt.scatter(range(len(raw_data)), raw_data, color='green', label='Raw', alpha=0.5)
    if len(mod_data) > 1:
        mod_data = np.concatenate(mod_data)
        plt.scatter(range(len(mod_data)), mod_data, color='blue', label='Modified', alpha=0.5)
    if len(nmod_data) > 1:
        nmod_data = np.concatenate(nmod_data)
        plt.scatter(range(len(nmod_data)), nmod_data, color='cyan', label='Not modified', alpha=0.5)
    if len(disc_data) > 1:
        disc_data = np.concatenate(disc_data)
        plt.scatter(range(len(disc_data)), disc_data, color='red', label='Discarded', alpha=0.5)
    plt.legend(loc=2)
    plt.title('Scatter Plot of Angles')
    plt.xlabel('Index')
    plt.ylabel('Angle')
    # Save the plots
    plt.savefig(os.path.join(plot_dir, f'{file}.png'))
    plt.close()

def plot_line_profiles(sampled_entries, plot_path, data_path, phase):
    """
    This function visualizes line profiles for a given set of sampled patches. 
    It generates a detailed figure with multiple subplots to analyze the patch data, 
    including individual line profiles for each pixel, combined line profiles, 
    label data, and a blended image showing the patch location within the original image. 
    The plots are saved to a directory specific to the phase (e.g., training or testing).
    """
    colormap = color_map()
    # Process each sampled entry to plot the data
    for entry in sampled_entries:
        nii_data, label_data, info_dict = entry
        image_file = info_dict['image']
        nature = info_dict['nature']
        h, w = info_dict['location'][:2]

        # Create a folder to save plots if it doesn't exist
        base_name = os.path.basename(image_file)
        main_folder = os.path.join(plot_path, os.path.splitext(base_name)[0])
        os.makedirs(main_folder, exist_ok=True)
        sample_folder = os.path.join(main_folder, phase)
        os.makedirs(sample_folder, exist_ok=True)

        # Load the original 2D image from the NIfTI file
        tif_image = imageio.imread(os.path.join(data_path,base_name.replace('stack.nii', 'dir_1.tif')))
        tiff_rgb = np.stack((tif_image,)*3, axis=-1)        
        
        # Create a single figure with 3 subplots
        fig, axes = plt.subplots(2, 2, figsize=(40, 40))
        fig.suptitle(f'Plots for Sample {os.path.basename(image_file)} - {nature} patch - Loc: {(h,w)}', size=40)

        # Subplot 1: Line Profiles by pixel
        profiles = nii_data.reshape(nii_data.shape)
        main_ax = axes[0, 0]
        grid = gridspec.GridSpecFromSubplotSpec(16, 16, subplot_spec=main_ax.get_subplotspec())
        for i in range(profiles.shape[0]):
            for j in range(profiles.shape[1]):
                ax = fig.add_subplot(grid[i, j])
                ax.plot(profiles[i, j, :])
                ax.grid(True)
                ax.set_xticks([])
                ax.set_yticks([])
        main_ax.set_xticks([])
        main_ax.set_yticks([])

        # Subplot 2: Line Profiles Combined
        for j in range(profiles.shape[0]):
            for k in range(profiles.shape[1]):
                axes[0,1].plot(profiles[j, k, :])
        axes[0, 1].grid()
        axes[0, 1].set_title('Line Profiles Combined')

        # Subplot 2: Label Data
        patch_height, patch_width = label_data.shape[0], label_data.shape[1]
        im2 = axes[1,0].imshow(label_data, cmap=colormap, vmin=0, vmax=180)
        axes[1,0].set_title(f"Label Patch", size=20)
        axes[1,0].set_xticks(np.arange(0.5, patch_height+0.5, 1))
        axes[1,0].set_xticklabels(np.arange(1, patch_width+1))
        axes[1,0].set_yticks(np.arange(0.5, patch_height+0.5, 1))
        axes[1,0].set_yticklabels(np.arange(1, patch_width+1))
        fig.colorbar(im2, ax=axes[1,0], fraction=0.05, pad=0.05)

        # Subplot 3: Blended Image with Patch Location Highlighted
        vis_patch = np.zeros((tif_image.shape[0], tif_image.shape[1], 3), dtype=np.uint8)
        vis_patch[h:h + patch_height, w:w + patch_width] = [255, 0, 0]
        
        blended_image = (0.7 * vis_patch + 0.7 * tiff_rgb).astype(np.uint8)
        
        axes[1,1].imshow(blended_image)  # Highlight patch in green
        axes[1,1].set_title('Patch Location', size=20)
        # Save the figure
        fig.tight_layout()
        plot_file = os.path.join(sample_folder, f"{os.path.basename(image_file)}_{nature}_line_profiles.png")
        plt.savefig(plot_file)
        plt.close()
      
      
#############    LSTM_CNN REGRESSION VISUALIZATION FUNCTIONS      #############

def plot_regression_results(y_true, y_pred, diff, epoch, plot_path, train):
    """
    This function visualizes regression results by plotting the true vs. predicted values 
    along with a regression line. It also includes a scatter plot of the predicted values against the true values 
    with a reference diagonal line. The plots are saved to the specified directory.
    Train flag ensures graphic analysis for training and validation datasets accordingly.
    """
    y_pred_adjusted = y_true + np.sign(y_pred - y_true) * diff
    # Fit a regression line to the adjusted predicted points
    model = LinearRegression().fit(np.arange(len(y_true)).reshape(-1, 1), y_pred_adjusted)
    
    # Plot the regression line
    line_x = np.linspace(0, len(y_true) - 1, 1000)
    line_y = model.predict(line_x.reshape(-1, 1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(np.arange(len(y_true)), y_true, alpha=0.6, edgecolors='none', s=10, c='blue', label='True Data')
    ax[0].scatter(np.arange(len(y_pred_adjusted)), y_pred_adjusted, alpha=0.6, edgecolors='none', s=10, c='red', label='Predicted Data')
    ax[0].plot(line_x, line_y, color='gray', label=f'Regression Line: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
    ax[0].set_title(f'Regression Plot - Epoch {epoch}')
    ax[0].set_xlabel('Pixel Index')
    ax[0].set_ylabel('Angle Value')
    ax[0].legend()
    ax[0].grid()
    ax[1].scatter(y_true, y_pred_adjusted, alpha=0.6)      
    ax[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2) #diagonal line for reference
    ax[1].set_xlabel('True Values')
    ax[1].set_ylabel('Predicted Values')
    ax[1].set_title('Regression Results')     
    ax[1].axis('equal')
    if train:
        plt.savefig(os.path.join(plot_path, 'train_regressor.png'))
    else:
        plt.savefig(os.path.join(plot_path, 'val_regressor.png'))
    plt.close()
    
def plot_regression_test(y_true, y_pred, diff, plot_path):
    """
    This function visualizes regression results by plotting the true vs. predicted values 
    along with a regression line. It also includes a scatter plot of the predicted values against the true values 
    with a reference diagonal line. The plots are saved to the specified directory.
    This version is for testing dataset
    """
    y_pred_adjusted = y_true + np.sign(y_pred - y_true) * diff
    # Fit a regression line to the adjusted predicted points
    model = LinearRegression().fit(np.arange(len(y_true)).reshape(-1, 1), y_pred_adjusted)
    
    # Plot the regression line
    line_x = np.linspace(0, len(y_true) - 1, 1000)
    line_y = model.predict(line_x.reshape(-1, 1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(np.arange(len(y_true)), y_true, alpha=0.6, edgecolors='none', s=10, c='blue', label='True Data')
    ax[0].scatter(np.arange(len(y_pred_adjusted)), y_pred_adjusted, alpha=0.6, edgecolors='none', s=10, c='red', label='Predicted Data')
    ax[0].plot(line_x, line_y, color='gray', label=f'Regression Line: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
    ax[0].set_title(f'Regression Plot')
    ax[0].set_xlabel('Pixel Index')
    ax[0].set_ylabel('Angle Value')
    ax[0].legend()
    ax[0].grid()
    ax[1].scatter(y_true, y_pred_adjusted, alpha=0.6)      
    ax[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2) #diagonal line for reference
    ax[1].set_xlabel('True Values')
    ax[1].set_ylabel('Predicted Values')
    ax[1].set_title('Regression Results')     
    ax[1].axis('equal')
    plt.savefig(os.path.join(plot_path, 'regressor_plot.png'))
    plt.close()
            
def plot_losses(train_losses, val_losses, epoch, plot_path):
    """
    This function plots the training and validation losses over epochs during model training.
    It creates a line plot showing how the losses change over time and saves the plot to the specified directory.
    """
    plt.figure()
    plt.plot(range(epoch + 1), train_losses, label='Training Loss')
    plt.plot(range(epoch + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_path, 'training_losses.png'))
    plt.close()
    
def plot_profiles(batch, idx, profiles, sample_folder):
    """
    This function generates and saves line profile plots for a specific batch and index. 
    It visualizes the profiles for each pixel in a 2D patch and saves the resulting plot in the specified folder.
    Is used on other functions below (e.g. compare_patches)
    """
    profiles_file = os.path.join(sample_folder, 'line_profiles.png')

    plt.figure(figsize=(20, 20))
    plt.suptitle(f'Line Profiles for Sample {batch}-{idx}', size=40)
    for i in range(profiles.shape[0]):
        for j in range(profiles.shape[1]):
            plt.subplot(profiles.shape[0], profiles.shape[1], i * profiles.shape[1] + j + 1)
            plt.plot(profiles[i, j, :])
            plt.grid()
            plt.title(f'Profile at ({i+1}, {j+1})')
    plt.tight_layout()
    plt.savefig(profiles_file)
    plt.close()

def compare_patches(true_labels, pred_labels, line_profiles, num_plots, plot_path, logger, loss_function):
    """
    This function compares true and predicted patches by visualizing them side by side. 
    It identifies the patches with the highest losses and generates plots for them, 
    including line profiles. The plots are saved in a directory specific to the batch and sample.
    """
    
    colormap = color_map()  # Function that creates the hsv_black colormap (see below)
    
    folder_path = os.path.join(plot_path, 'predicted_patches')
    os.makedirs(folder_path, exist_ok=True)
    
    for i in range(len(true_labels)):  # Iterate over each batch
        if i < 4:   
            logger.info(f"Plotting quick comparison for batch {i}..")
            batch_folder = os.path.join(folder_path, f'batch_{i}')
            os.makedirs(batch_folder, exist_ok=True)

            true_patches = true_labels[i].cpu().numpy()  # Shape: [b, w, h]
            pred_patches = pred_labels[i].cpu().numpy()
            profiles = line_profiles[i].cpu().numpy()
                
            # Calculate loss for each patch
            losses = []
            for j in range(true_patches.shape[0]):
                true_patch = torch.tensor(true_patches[j])
                pred_patch = torch.tensor(pred_patches[j])
                loss = loss_function(pred_patch, true_patch).item()
                losses.append(loss)
                
            # Select indices with highest losses
            sorted_indices = np.argsort(losses)[-num_plots:]    # This determines the biggest loss patches for plotting purposes. This can be changable for random tracking.

            for j, idx in enumerate(sorted_indices):
                sample_folder = os.path.join(batch_folder, f'sample_{idx}')
                os.makedirs(sample_folder, exist_ok=True)

                true_patch = mask(true_patches[idx])  # Apply color mask of -1
                pred_patch = mask(pred_patches[idx])

                plt.figure(figsize=(15, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(true_patch, cmap=colormap, vmin=0, vmax=180)
                plt.title(f"True Labels Patch {i}-{idx}")
                plt.xticks(np.arange(0.5, 8.5, 1),np.arange(1, 9))
                plt.yticks(np.arange(0.5, 8.5, 1),np.arange(1, 9))
                plt.subplot(1, 2, 2)
                plt.imshow(pred_patch, cmap=colormap, vmin=0, vmax=180)
                plt.title(f"Predicted Labels Patch {i}-{idx}")
                plt.xticks(np.arange(0.5, 8.5, 1),np.arange(1, 9))
                plt.yticks(np.arange(0.5, 8.5, 1),np.arange(1, 9))
                plt.tight_layout()
                plt.savefig(os.path.join(sample_folder, f"angles.png"))
                plt.close()

                plot_profiles(i, idx, profiles[idx], sample_folder)
                
def plot_all_train_losses(all_train_losses, all_val_losses, datasets, plot_path):
    # Plot all the results together for comparison across folds
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        ax[0].plot(train_losses, label=f'{datasets[i]} Train Loss')
        ax[1].plot(val_losses, label=f'{datasets[i]} Val Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Losses Across Datasets')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Validation Losses Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'cv_dataset_losses.png'))
    plt.close()

def plot_all_test_metrics(all_losses, all_r2, datasets, test_path):
    # Plot all the results together (this will be edited later and added to visualization.py file)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_ticks = list(range(len(datasets)))
    ax[0].plot(all_losses, marker='o')
    ax[0].set_xticklabels(datasets, rotation=90)
    ax[0].set_xlabel('Datasets')
    ax[0].set_ylabel('Final Loss')
    ax[0].set_title('Test Losses All Datasets')
    ax[0].grid(True)
    ax[1].set_ticks = list(range(len(datasets)))
    ax[1].plot(all_r2, marker='o')
    ax[1].set_xticklabels(datasets, rotation=90)
    ax[1].set_xlabel('Datasets')
    ax[1].set_ylabel('Final r2')
    ax[1].set_title('Test r2 All Datasets')
    ax[1].grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(test_path, 'cv_test_metrics.png'))
    plt.close()
    
                    
#############    GAN REGRESSION VISUALIZATION FUNCTIONS      #############

def plot_lstm_losses(train_true_losses, val_true_losses, epoch, plot_path):
    """
    This function plots the training and validation losses for an LSTM model over the epochs.
    It generates a line plot showing how the losses change over time and saves the plot as an image file 
    in the specified directory.
    """
    plt.figure(figsize=[5,5])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(epoch + 1), train_true_losses, label='Training')
    plt.plot(range(epoch + 1), val_true_losses, label='Validation')
    plt.title('LSTM Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_path, 'lstm_losses.png'))
    plt.close()
    
def plot_gan_losses(gen_train_losses, adv_train_losses, gen_val_losses, disc_train_real_losses, disc_train_fake_losses, epoch, plot_path):
    """
    This function plots the losses associated with training a GAN (Generative Adversarial Network).
    It produces three subplots: one for the generator's MSE loss, one for the adversarial BCE loss, 
    and one for the discriminator's BCE loss on real and fake labels. These plots are saved as an image file 
    in the specified directory.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax[0].plot(range(epoch + 1), gen_train_losses, label='Training')
    ax[0].plot(range(epoch + 1), gen_val_losses, label='Validation')
    ax[0].set_title('Generator MSE Training and Validation Loss')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(range(epoch + 1), adv_train_losses, label='Training')
    ax[1].set_title('Adversarial BCE Generator Training Loss')
    ax[1].legend()
    ax[1].grid(True)
    ax[2].plot(range(epoch + 1), disc_train_real_losses, label='Training')
    ax[2].plot(range(epoch + 1), disc_train_fake_losses, label='Validation')
    ax[2].set_title('Discriminator BCE Training Real and Fake Labels Loss')
    ax[2].legend()
    ax[2].grid(True)
    plt.savefig(os.path.join(plot_path, 'gan_losses.png'))
    plt.close()    

def plot_patches(output, target, epoch, plot_path, model_name, train):
    """
    This function visualizes and compares the predicted and true label patches for a model during training or testing.
    It plots the true labels and predicted labels side by side and saves the images in the specified directory.
    This function is particularly useful for observing how well the model is performing on a per-patch basis.
    """
    os.makedirs(os.path.join(plot_path, model_name), exist_ok=True)
    colormap = color_map()
    output, target = output.detach().cpu(), target.detach().cpu()
    output, target = output[0,:,:], target[0,:,:]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    plt.suptitle(f'Epoch {epoch+1}', fontsize=16, weight='bold')
    if train:
        im0 = ax[0].imshow(target, cmap=colormap, vmin=-1, vmax=180)
        ax[0].set_title(f'True Training Label')
        ax[0].set_xticks(np.arange(0.5, 16.5, 1))
        ax[0].set_xticklabels(np.arange(1, 17))
        ax[0].set_yticks(np.arange(0.5, 16.5, 1))
        ax[0].set_yticklabels(np.arange(1, 17))
        im1 = ax[1].imshow(output, cmap=colormap, vmin=-1, vmax=180)
        ax[1].set_title(f'Predicted Training {model_name} label')
        ax[1].set_xticks(np.arange(0.5, 16.5, 1))
        ax[1].set_xticklabels(np.arange(1, 17))
        ax[1].set_yticks(np.arange(0.5, 16.5, 1))
        ax[1].set_yticklabels(np.arange(1, 17))
        fig.colorbar(im1, ax=ax[1], fraction=0.05, pad=0.05)
    else:
        im2 = ax[2].imshow(target, cmap=colormap, vmin=-1, vmax=180)
        ax[2].set_title('True Training Label')
        ax[2].set_xticks(np.arange(0.5, 16.5, 1))
        ax[2].set_xticklabels(np.arange(1, 17))
        ax[2].set_yticks(np.arange(0.5, 16.5, 1))
        ax[2].set_yticklabels(np.arange(1, 17))
        im3 = ax[3].imshow(output, cmap=colormap, vmin=-1, vmax=180)
        ax[3].set_title(f'Predicted {model_name} label')
        ax[3].set_xticks(np.arange(0.5, 16.5, 1))
        ax[3].set_xticklabels(np.arange(1, 17))
        ax[3].set_yticks(np.arange(0.5, 16.5, 1))
        ax[3].set_yticklabels(np.arange(1, 17))
        fig.colorbar(im2, ax=ax[3], fraction=0.05, pad=0.05)
    plt.tight_layout()        
    plt.savefig(os.path.join(plot_path, model_name, 'label_prediction.png'))
    plt.close()


#############    FINAL IMAGE COMPARISON VISUALIZATION FUNCTIONS      #############

def make_fom(angle_patches, locations, image_shape, patch_size):
    """
    Create a full FOM image by placing angle patches at specified locations.
    Args:
        angle_patches: List of angle patches to be placed in the FOM.
        locations: List of (y, x) locations indicating where each patch should be placed in the FOM.
        image_shape: Tuple indicating the shape (height, width) of the full image.
        patch_size: Size of the patches to be placed in the image.
    Returns:
        fom_image: The reconstructed FOM image with all patches placed at their corresponding locations.
    """
    # Create an empty image
    fom_image = np.full((image_shape[0], image_shape[1]), fill_value=-1.0, dtype=np.float32)

    # Place the patches in the image at the specified locations
    for (y, x), angle_patch in zip(locations, angle_patches):
        fom_image[x:x+patch_size, y:y+patch_size] = angle_patch

    return fom_image

def improved_fom(fom_output, fom_target):
    """
    Improve the target FOM by filling in missing or invalid values with the predicted values.
    Args:
        fom_output: The predicted FOM image.
        fom_target: The target FOM image containing some invalid (-1) values.
    Returns:
    """  
    improved_fom = fom_target
    for i in range (improved_fom.shape[0]):
        for j in range (improved_fom.shape[1]):
            if improved_fom[i, j] == -1:
                improved_fom[i, j] = fom_output[i, j]
    return improved_fom

def transform_image(img):
    """
    Transform the image (rotation and flipp) as in the preprocessing step.
    """
    #img = np.array(img, dtype=np.float32)
    squeezed_img = np.squeeze(img)
    flipped_img = np.flip(squeezed_img, axis=1)
    transformed_img = np.rot90(flipped_img, k=3, axes=(1, 0))
    return transformed_img
            
def compare_test_images(all_targets, all_outputs, all_dicts, plot_path, patch_size, logger):
    """
    The function is designed to compare the predicted and true images FOMs across all test images. 
    It generates visualizations that highlight differences between the true and predicted images, as well as improvements made by the model.
    The function also saves these visualizations and outputs the results as images in the specified directory.
    """
    colormap = color_map()  # Define the colormap for visualizing the images
    colormap2 = color_map2()  # Define a secondary colormap for differences
    
    folder_path = os.path.join(plot_path, 'predicted_images')  # Create directory to save predicted images
    os.makedirs(folder_path, exist_ok=True) 
    unique_images = {}  # Dictionary to keep track of unique images and their dimensions
    
    # Identify unique images and their dimensions
    for i in all_dicts:
        if i['image'] not in unique_images:
            unique_images[i['image']] = (3272, 2469)  # Assuming fixed image dimensions
    
    # Process each unique image
    for image, image_shape in unique_images.items():
        loc = []  # List to store locations of patches
        patches_output = []  # List to store output patches
        patches_target = []  # List to store target (true) patches
        
        # Iterate over all outputs, targets, and metadata dictionaries
        for output, target, meta_dict in zip(all_outputs, all_targets, all_dicts):
            #output = transform_image(output)      # uncomment this if needed (LSTM_CNN_GAN train loop already has this step, but ont LTSM_CNN)
            #target = transform_image(target)      # uncomment this if needed (LSTM_CNN_GAN train loop already has this step, but ont LTSM_CNN)
                
            loc_y, loc_x = meta_dict['location'][0], meta_dict['location'][1]  # Extract patch location
            if meta_dict['image'] == image:  # Check if the patch belongs to the current image
                loc.append((loc_y, loc_x))  # Store location
                patches_output.append(output)  # Store output patch
                patches_target.append(target)  # Store target patch
        
        logger.info(f"Plotting {image} real and predicted FOMs..")  # Log the current processing image
        
        # Create full images (FOMs) from patches
        fom_output = make_fom(patches_output, loc, image_shape, patch_size)  # Predicted FOM
        fom_target = make_fom(patches_target, loc, image_shape, patch_size)  # True FOM
        fom_improved = improved_fom(fom_output, fom_target)  # Improved FOM based on prediction
        fom_difference = fom_target - fom_output  # Difference between true and predicted FOMs

        # Save the FOM images
        im_out = Image.fromarray(fom_output)
        im_out.save(os.path.join(folder_path, 'fom_output.tiff'))
        im_tar = Image.fromarray(fom_target)
        im_tar.save(os.path.join(folder_path, 'fom_target.tiff'))
        im_imp = Image.fromarray(fom_improved)
        im_imp.save(os.path.join(folder_path, 'fom_improved.tiff'))
        im_imp = Image.fromarray(fom_difference)
        im_imp.save(os.path.join(folder_path, 'fom_dif.tiff'))
        
        # Plot side-by-side comparison of FOMs
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))
        
        im1 = axes[0].imshow(fom_target, cmap=colormap, vmin=0, vmax=181)
        axes[0].set_title(f"Real Angle FOM - {image}")
        axes[0].axis('off')
        
        im2 = axes[1].imshow(fom_output, cmap=colormap, vmin=0, vmax=181)
        axes[1].set_title(f"Predicted Angle FOM - {image}")
        axes[1].axis('off')
        
        im3 = axes[2].imshow(fom_difference, cmap=colormap2, vmin=0, vmax=181)
        axes[2].set_title(f"FOM Difference Map - {image}")
        axes[2].axis('off')
        
        im4 = axes[3].imshow(fom_improved, cmap=colormap, vmin=0, vmax=181)
        axes[3].set_title(f"Improved Predicted FOM - {image}")
        axes[3].axis('off')

        # Add colorbars for each subplot
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

        # Save the comparison plot
        plot_file_path = os.path.join(folder_path, f'comparison_{os.path.basename(image)}.png')
        plt.savefig(plot_file_path)
        plt.close(fig)
        
    logger.info(f"All plots saved in {plot_path}")  # Log that all plots have been saved


#############    OTHER VISUALIZATION FUNCTIONS      #############
                   
def color_map():                            ### Colormap imitating hsv_black
    num_steps = 180  # Total steps needed for range 0 to 179
    hues = np.linspace(0, 1, num_steps, endpoint=False)  # Full HSV range from 0 to 1
    hsv = np.stack((hues, np.ones(num_steps), np.ones(num_steps)), axis=1)
    rgb = hsv_to_rgb(hsv)
    color_map = np.vstack([[0, 0, 0], rgb])  # Set the first color to black
    return ListedColormap(color_map)

def color_map2():                            ### Colormap for plotting diference maps
    # Generate the 'YlOrBr' colormap
    base_cmap = get_cmap('YlOrBr', 181)  # 181 colors for 1 to 180
    color_list = base_cmap(np.linspace(0, 1, 181))
    new_color_list = np.vstack([[0, 0, 0, 1], color_list[1:]])  # RGBA for black at the start
    custom_cmap = ListedColormap(new_color_list)
    return custom_cmap

def mask(patch):                          ## Mask function (invalid pixels)
    masked_patch = patch.copy()
    # Set values less than 0 or greater than 180 to 0 (black)
    masked_patch[(patch < 0) | (patch > 180)] = 0
    # Increment values within the range [0, 180] by 1 to map them correctly
    masked_patch[(patch >= 0) & (patch <= 180)] += 1
    return masked_patch