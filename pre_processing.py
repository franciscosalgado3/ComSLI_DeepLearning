import numpy as np                           # Importing numpy for numerical operations on arrays
import nibabel as nib                        # Importing nibabel to handle neuroimaging data files (like NIfTI)
from PIL import Image, ImageEnhance          # Importing PIL (Pillow) to handle image files
from loggs import logger                     # Importing the centralized logger from loggs.py
import random                                # Importing random for generating random selections
from sklearn.model_selection import KFold    # Importing KFold for cross-validation
import os

from config import paths                                                        # Import paths from the configuration file
from visualization import plot_raw_data, plot_all_patches, plot_occurrences, plot_line_profiles     # functions from visualization.py module to handle different types of data plotting

class Load_n_Normalize():
    """
    This functions load the images and labels, checks voxel sizes of images, and normalizes the stack image. Returns training and testing data in different variables
    """

    def __init__(self, train_image_paths: list, train_label_paths: list, test_image_paths: list, test_label_paths: list, plot_path: str):
        self.train_image_paths = train_image_paths      # List of paths to training images
        self.train_label_paths = train_label_paths      # List of paths to training labels
        self.test_image_paths = test_image_paths        # List of paths to test images
        self.test_label_paths = test_label_paths        # List of paths to test labels
        self.plot_path = plot_path                      # Path where plots will be displayed
        self.logger = logger                            # Reassign logger in case it's needed within this subclass

    def _load_image(self, img_path):
        """
        Load and preprocess a neuroimaging file (NIfTI format)
        """
        
        img = nib.load(img_path)
        transformed_img = np.rot90(np.flip(np.squeeze(np.array(img.get_fdata())), axis=1), k=3, axes=(1, 0))
        return transformed_img

    def _load_label(self, lab_path):
        """
        Load a label image (TIFF format)
        """
        
        label = np.array(Image.open(lab_path))
        return label
    
    def _check_voxel_size(self, images):
        """
        Check and log if all images have the same voxel size
        """
        
        self.logger.info("Checking voxel sizes...")
        initial_voxel_size = None
        for img_path in images:
            voxel_size = nib.load(img_path).header.get_zooms()[:3]
            if initial_voxel_size is None:
                initial_voxel_size = voxel_size
                self.logger.info(f"Initial voxel size set to {initial_voxel_size}")
            elif voxel_size != initial_voxel_size:
                self.logger.warning(f"Voxel size mismatch found in file {img_path}, expected {initial_voxel_size}, got {voxel_size}")

    def _normalize_stack(self, image):
        """
        Normalize an image stack to the range [0, 1]
        """
        
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized_image
    
    def preprocess(self):
        """
        Main method to load and normalize training and testing images and labels
        """
        
        self.logger.info(f"Loading training objects and normalizing images stack-wise. Plotting raw data..")
        
        self._check_voxel_size(self.train_image_paths)         # Check that all images have the same voxel size
        processed_train_data = []
        for img_path, lab_path in zip(self.train_image_paths, self.train_label_paths):
            image = self._load_image(img_path)
            label = self._load_label(lab_path)
            normalized_image = self._normalize_stack(image)
            file_name = img_path.split('/')[-1]          # Extracts the file name from the path
            plot_raw_data(normalized_image, label, file_name, self.plot_path, train_flag = True)   # Plot raw training data             
            processed_train_data.append((file_name, normalized_image, label))   #Savind data as list where each entry is a tuple contanining (file_name, normalized_image, label)
              
        self.logger.info(f"Loading tetsing objects and normalizing images stack-wise. Plotting raw data..")
        
        self._check_voxel_size(self.test_image_paths)         # Check that all images have the same voxel size
        processed_test_data = []
        for img_path, lab_path in zip(self.test_image_paths, self.test_label_paths):     
            image = self._load_image(img_path)
            label = self._load_label(lab_path)
            normalized_image = self._normalize_stack(image)
            file_name = img_path.split('/')[-1]          # Extracts the file name from the path
            plot_raw_data(normalized_image, label, file_name, self.plot_path, train_flag = False)    # Plot raw tetsing data 
            processed_test_data.append((file_name, normalized_image, label))    #Savind data as list where each entry is a tuple contanining (file_name, normalized_image, label)
            
        return processed_train_data, processed_test_data     # Return the processed training and testing data


class Patch_Processor():
    """
    This function processes patches from the microresulotion training and testing images, assiginng metadata for monitoring the patches accordingly
    """
    
    def __init__(self, patch_size: int, allow_modification: bool, neighbors: int, plot_path: str, num_profile_plots: int, stride: int):
        self.patch_size = patch_size                        # Size of the patches to be extracted
        self.stride = stride                                # Stride to use when extracting patches
        self.allow_modification = allow_modification        # Whether or not to allow modification of patches (see function below)
        self.neighbors = neighbors                          # Number of neighbors to consider in patch modification (see function below)
        self.num_profile_plots = num_profile_plots          # Number of line profile plots to generate (to monitor pre-processing)
        self.plot_path = plot_path                          # Path where plots will be displayed
        self.logger = logger                                # Reassign logger in case it's needed within this subclass


    def _average_4_neighbors(self, tiff_image, y, x):
        """
        Calculate the average of 4 neighbors for a given pixel in a TIFF image
        """
        
        neighbors = []
        # Check 4 neighbors in cross pattern
        for ny, nx in [(y-1, x), (y, x-1), (y, x+1), (y+1, x)]:
            if 0 <= ny < tiff_image.shape[0] and 0 <= nx < tiff_image.shape[1] and tiff_image[ny, nx] != -1:
                neighbors.append(tiff_image[ny, nx])
        return neighbors
    
    def _average_8_neighbors(self, tiff_image, y, x):
        """
        Calculate the average of 8 neighbors for a given pixel in a TIFF image
        """
        
        neighbors = []
        current_value = tiff_image[y, x]
        # Check 8 neighbors including diagonals
        for ny, nx in [(y-1, x), (y-1, x-1), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x), (y+1, x-1), (y+1, x+1)]:
            if 0 <= ny < tiff_image.shape[0] and 0 <= nx < tiff_image.shape[1]:
                neighbor_value = tiff_image[ny, nx]
                if neighbor_value != -1:
                    # Calculate the distance modulo 180 degrees
                    distance = abs(current_value - neighbor_value) % 180
                    if distance <= 30 or distance >= 150:  # Because 180 - 30 = 150, if distance is below 30º is considered
                        neighbors.append(neighbor_value)
        return neighbors
    
    def _modify_patch(self, tiff_image):
        """
        Modify the patch by filling missing values based on the number of neighbors and the method choosed
        """
        
        modified_image = tiff_image.copy()
        for y in range(tiff_image.shape[0]):
            for x in range(tiff_image.shape[1]):
                if tiff_image[y, x] == -1:
                    #num_neighbors = self._average_4_neighbors(tiff_image, y, x)
                    num_neighbors = self._average_8_neighbors(tiff_image, y, x)    #choose the bestion option
                    if len(num_neighbors) >= self.neighbors:
                        #modified_image[y, x] = np.mean(num_neighbors)
                        modified_image[y, x] = np.median(num_neighbors)        #again choose the best option
        return modified_image

    def process_dataset(self, train_data, test_data):
        """"
        Main method to process training and testing datasets independently. Include all patches regardless of their initial quality.
        """
        
        all_train_data = []
        self.logger.info(f"Pre-processing training dataset...")
        for (file, nii_data, tiff_data) in train_data:          # for each training image file found, process patches 
            self.logger.info(f"Defining patches for {file} image and plotting results...")
            processed_data = self._process_all_patches(file, nii_data, tiff_data, train_flag = True)
            all_train_data.extend(processed_data)
            
        all_test_data = []
        self.logger.info(f"Pre-processing testing dataset...")
        for (file, nii_data, tiff_data) in test_data:          # for each testing image file found, process patches 
            self.logger.info(f"Defining patches for {file} image..")
            processed_data = self._process_all_patches(file, nii_data, tiff_data, train_flag = False)
            all_test_data.extend(processed_data)
                    
        self.logger.info(f"Setting {self.num_profile_plots} train and test patches to monitor the patche location and line profiles shape...")
        
        # For monitoring line profiles location, shape and label, this function _set_plot_entries was created to define some patches to be plotted and followed across the dataset pipeline
        train_prof_samples = self._set_plot_entries(all_train_data, self.num_profile_plots)
        test_prof_samples = self._set_plot_entries(all_test_data, self.num_profile_plots)
    
        self.logger.info(f"Patches selection complete.")
        # returns 4 lists each one containing a triple tuple in each entry with (patch image, patch label, patch metada); train_prof_samples, test_prof_samples  is to monitor patches
        return all_train_data, all_test_data, train_prof_samples, test_prof_samples   
    
    
    def _process_all_patches(self, file, nii_data, tiff_data, train_flag):
        """
        This function processes all possible patches from the dataset using a sliding window approach, assigning patch metadata like location, type, nature
        """
        
        # Check if the spatial dimensions of the NIfTI data and the TIFF data match
        if nii_data.shape[:2] != tiff_data.shape:
            raise ValueError("NIfTI and TIFF data have incompatible spatial dimensions.")
        
        # Initialize an empty RGB image array for label visualization purposes.
        vis_image = np.zeros((tiff_data.shape[0], tiff_data.shape[1], 3), dtype=np.uint8)
        
        # Calculate the step size for the sliding window based on the patch size and stride
        step = self.patch_size - self.stride
        
        # Initialize an empty list to store the processed patches and their associated metadata
        data = []

        # Loop over the image to extract patches using the sliding window approach
        for y in range(0, nii_data.shape[0] - self.stride, step):
            for x in range(0, nii_data.shape[1] - self.stride, step):
                # Skip this patch if it would extend beyond the image boundaries
                if y + self.patch_size > nii_data.shape[0] or x + self.patch_size > nii_data.shape[1]:
                    continue
                
                # Extract the current patch from the NIfTI data
                nii_patch = nii_data[y:y + self.patch_size, x:x + self.patch_size, :]
                
                # Extract the corresponding patch from the TIFF data
                tiff_patch = tiff_data[y:y + self.patch_size, x:x + self.patch_size]

                # Check if the TIFF patch (label) does not have the value 200 (indicating it's relevant data). This wil not be considered for training and testing.
                # This was the value given in Fiji when processing the labels to mask irrelevant data (different then 1D, so background and 2 fiber region).
                # So, the label raw values are:
                # -1 -> not labeled pixels
                # 0-180 -> angle value orientation
                # 200 -> invalid pixel
                if np.all(tiff_patch != 200):
                    
                    # If the patch has no missing labels (-1), consider it "raw" data
                    if -1 not in tiff_patch:
                        if train_flag:  # If this is part of the training set
                            data.append((nii_patch, tiff_patch, {'image': file, 'location': (y, x), 'type': 'training', 'nature': 'raw'}))   #triple tuple with (image patch, label patch, patch metadata)
                        else:  # If this is part of the testing set
                            data.append((nii_patch, tiff_patch, {'image': file, 'location': (y, x), 'type': 'testing', 'nature': 'raw'}))
                        # Mark this patch as "raw" with green color in the visualization
                        vis_image[y:y + self.patch_size, x:x + self.patch_size] = [0, 255, 0]  # Green for raw
                    
                    # Handle patches that contain missing data (-1)
                    else:
                        if train_flag:    # If this is part of the training set
                            if np.all(tiff_patch == -1):  # If the patch is all -1 values, skip this patch
                                continue
                            if self.allow_modification:  # If modifications to patches are allowed
                                modified_patch = self._modify_patch(tiff_patch)
                                # If the patch was modified, append it to the data as "modified"
                                if np.any(modified_patch != tiff_patch):
                                    data.append((nii_patch, modified_patch, {'image': file, 'location': (y, x), 'type': 'training', 'nature': 'modified'}))
                                    vis_image[y:y + self.patch_size, x:x + self.patch_size] = [0, 0, 255]  # Blue for modified
                                else:  # If the patch was not modified, consider it "not modified"
                                    data.append((nii_patch, tiff_patch, {'image': file, 'location': (y, x), 'type': 'training', 'nature': 'not modified'}))
                                    vis_image[y:y + self.patch_size, x:x + self.patch_size] = [255, 0, 0]  # Red for not modified
                            else:  # If modifications are not allowed, consider it "not modified"
                                data.append((nii_patch, tiff_patch, {'image': file, 'location': (y, x), 'type': 'training', 'nature': 'not modified'}))
                                vis_image[y:y + self.patch_size, x:x + self.patch_size] = [255, 0, 0]  # Red for not modified
                        else:  # If this is part of the testing set
                            data.append((nii_patch, tiff_patch, {'image': file, 'location': (y, x), 'type': 'testing', 'nature': 'not modified'}))
                            vis_image[y:y + self.patch_size, x:x + self.patch_size] = [255, 0, 0]  # Red for not modified
                                
        # Plot all patches (both "good" and "bad") for visualization
        plot_all_patches(tiff_data, vis_image, data, file, self.plot_path, train_flag)
        
        return data    # Return the processed patch data as a list

    
    def _set_plot_entries(self, data, num_plots):
        """
        Save a number of diferent entries entries (image, label, metadata) according to `num_plots`.
        E.g.if num_plots = 2 save 2 entries per nature (raw, modified, not modified) for each image file.
        The ideia is to save samples to visualiza it later, during the dataset pipeline, to see check differences
        in the line profiles in different datasets, as well as location of the patch and label layout.
        """
        sampled_entries = []  # List to store the selected entries
        image_dict = {}       # Dictionary to categorize entries by image file and nature

        # Iterate over the data to organize it by image file and nature (e.g., raw, modified, not modified)
        for i, entry in enumerate(data):
            _, _, info_dict = entry           # Unpack the entry; we are interested in the metadata (info_dict)
            image_file = info_dict['image']    # Get the name of the image file from the metadata
            if image_file not in image_dict:
                image_dict[image_file] = {}      # Initialize a nested dictionary for the image file
            nature = info_dict['nature']           # Get the nature (raw, modified, not modified) from the metadata
            if nature not in image_dict[image_file]:
                image_dict[image_file][nature] = []     # Initialize a list to store indices for this nature
            image_dict[image_file][nature].append(i)      # Store the index of this entry under the corresponding nature

        # Now, for each image file and each nature, select up to `num_plots` entries to sample
        for image_file, nature_dict in image_dict.items():
            for nature, indices in nature_dict.items():
                if len(indices) >= num_plots:
                    sampled_indices = random.sample(indices, num_plots)  # Randomly sample `num_plots` indices
                else:
                    sampled_indices = indices  # If there aren't enough indices, just use all available

                # Add the sampled entries to the list of sampled entries
                for index in sampled_indices:
                    sampled_entries.append(data[index])

        return sampled_entries  # Return the list of sampled entries
    

class Augment_Dataset():
    """
    This class ensures data augmentation to balance the dataset in orientations and give more data variability.
    It also creates different datasets based on different noise adiction on line profiles.
    The parameters are tunnable in order to find optimal values for training to mimic noisy testing line profiles.
    """
    def __init__(self, augment_ratio: float, plot_path: str, k_folds: int, g_mean: float, g_std: float, s_mean: float, s_std: float, cont_factor: float, bright_factor: float, max_slices: int, lower_value: float, higher_value: float):
        """
        Initialize the dataset with parameters for data augmentation and dataset processing
        """
        
        self.augment_ratio = augment_ratio          # This ratio decide the level of augmentation (of all patches)
        self.plot_path = plot_path                  # Path where plots will be displayed
        self.k_folds = k_folds                      # Number of folds(datasets) for cross-validation
        self.logger = logger                        # Assign the centralized logger
        self.g_mean = g_mean                        # Mean for Gaussian noise
        self.g_std = g_std                          # Standard deviation for Gaussian noise
        self.s_mean = s_mean                        # Mean for speckle noise
        self.s_std = s_std                          # Standard deviation for speckle noise
        self.bright_factor = bright_factor          # Factor for adjusting brightness
        self.cont_factor = cont_factor              # Factor for adjusting contrast
        self.max_slices = max_slices                # Maximum number of slices to apply customized noise
        self.higher_value = higher_value            # Upper bound for customized noise
        self.lower_value = lower_value              # Lower bound for customized noise
    
    def _shuffle_data(self, data):
        """
        Shuffle the dataset to randomize the order of data points
        """
        
        indices = np.arange(len(data))  # Create an array of indices based on the length of the data
        np.random.shuffle(indices)       # Randomly shuffle these indices
        data_tuple = [data[i] for i in indices]  # Reorder data based on shuffled indices
        
        # Extract elements into separate lists for images, labels, and metadata
        shuff_images = [item[0] for item in data_tuple]
        shuff_labels = [item[1] for item in data_tuple]
        shuff_dict = [item[2] for item in data_tuple]

        return shuff_images, shuff_labels, shuff_dict  # Return shuffled images, labels, and metadata

    def _calculate_thresholds(self, occurrences):
        """
        Calculate the threshold for augmenting and balancing the dataset based on the total occurrence of patches.
        Gives threshold value for _augment_patches function. 
        This parameter can be tunnable but values bigger than ~0.8 can lead into errors because of randomusing too many samples for augmentation
        """
        
        total_patches = sum(occurrences.values())                                     # Sum the occurrences of all patches
        min_occurrences = int(self.augment_ratio * total_patches / len(occurrences))  # Calculate the minimum occurrences required based on augmentation ratio
        threshold = min_occurrences + 1                                               # Set the threshold slightly above the minimum to avoid errors. This can also be tunned.
        return min_occurrences, threshold                        # Return the calculated minimum occurrences and threshold
    
    def _patch_mean(self, labels, list_dict):
        """
        Calculate the mean angle for each patch to use in _augment_patches function.
        Creates a dictionary with the occurrences with the mean angle values of each patch [0,180] in the dataset.
        """
        
        occurrences = {}             # Initialize an empty dictionary to track occurrences
        for i in range(len(labels)):
            label = labels[i]              # Get the label for the current patch
            mask = labels[i] != -1           #  Create a mask to ignore invalid pixels in the count
            mean_angle = int(np.floor(np.mean(label[mask]) % 180))  # Calculate the mean angle of valid pixels in the range [0,180]
            list_dict[i]['mean_angle_patch'] = mean_angle              # Store the mean angle in the metadata
            occurrences[mean_angle] = occurrences.get(mean_angle, 0) + 1  # Update occurrences of this mean angle
        return labels, list_dict, occurrences                              # Return the updated labels, metadata, and occurrences
    
    def _augment_patches(self, data_tuple):
        """
        Augment patches to ensure all angles have sufficient representation.
        Patches with more representation than the threshold value will be used to simulate other occurences that have lower values than min_occurences (until reaching this value).
        The stack is rolled taking in mind that rolling a x number of slices is incrementing a x multiplier of 15º - another patch and label are created randomly.
        each augmented patch is just once used for this purpose, its not re-used.
        Occurences dict are updated and histogram is plotted before and after this process.
        """
        
        patches, labels, list_dict = self._shuffle_data(data_tuple)           # Shuffle the data
        labels, list_dict, occurrences = self._patch_mean(labels, list_dict)  # Calculate mean angles and occurrences
        min_occurrences, threshold = self._calculate_thresholds(occurrences)  # Calculate thresholds for augmentation
        
        self.logger.info(f"Plotting patches occurrences before dataset augmentation and balancing..")
        plot_occurrences(occurrences, min_occurrences, threshold, 'before_aug', self.plot_path)  # Plot occurrences before augmentation
        aug_patch_indices = set()  # Track indices of augmented patches
        
        self.logger.info(f"Start data augmentation taking into account a ratio of {self.augment_ratio} and a threshold of {threshold} patches per angle occurrence..")
        # Continue augmenting patches until all angles meet the minimum occurrence threshold
        while any(count < min_occurrences for count in occurrences.values()):
            for i in range(len(labels)):
                if i in aug_patch_indices:  # Skip patches that have already been augmented
                    continue
                
                mask = labels[i] != -1  # Mask to ignore invalid pixels
                data_patch = patches[i]
                label_patch = labels[i]
                mean_angle = int(np.floor(np.mean(label_patch[mask]) % 180))  # Calculate mean angle for the current patch

                if occurrences.get(mean_angle, 0) > threshold:  # If the angle exceeds the threshold, augment it
                    multiplier = random.randint(1, 12)                # Randomly choose a multiplier for augmentation
                    augmented_angle = (mean_angle + multiplier * 15) % 180  # Calculate the new augmented angle

                    if occurrences.get(augmented_angle, 0) < threshold:      # If the augmented angle is underrepresented
                        new_label_patch = (label_patch + multiplier * 15) % 180  # Adjust the label patch by the multiplier
                        new_label_patch = np.where(mask, new_label_patch, -1)       # Apply the mask to maintain valid pixel positions
                        labels[i] = new_label_patch  # Update the label patch

                        new_data_patch = np.roll(data_patch, -multiplier, axis=2)  # Roll the data patch to simulate rotation
                        patches[i] = new_data_patch  # Update the data patch

                        occurrences[augmented_angle] = occurrences.get(augmented_angle, 0) + 1  # Update occurrences for augmented angle
                        occurrences[mean_angle] = occurrences.get(mean_angle, 0) - 1  # Decrease occurrences for the original angle
                        
                        list_dict[i]['mean_angle_patch'] = augmented_angle  # Update metadata with the new augmented angle
                        list_dict[i]['nature'] = 'augmented'            # Mark the patch as augmented
                        aug_patch_indices.add(i)                      # Track this index as augmented
        
        self.logger.info(f"Plotting patches occurrences after dataset augmentation...")   
        plot_occurrences(occurrences, min_occurrences, threshold, 'after_aug', self.plot_path)  # Plot occurrences after augmentation
        
        # Return the augmented patches, labels, and updated occurrences
        return patches, labels, list_dict, occurrences, min_occurrences, threshold

    def balance_train_data(self, data_tuple):
        """
        Balance the training data by ensuring angle occurrences are within a specific range - creating a balanced base-dataset of angle representation.
        Minimum value is secured by data augmentation of patches (_augment_patches).
        Maximum value is secured by randomly discard patches.
        """
        
        patches, labels, list_dict, occurrences, min_occurrences, threshold = self._augment_patches(data_tuple)    # Augment patches first        
        disc_index = []                 # Initialize a list to track indices of discarded patches
        data = []                           # Initialize a list to hold the balanced dataset
        max_occur = min_occurrences + 15       # Set the maximum allowed occurrence per angle
        self.logger.info(f"Balancing training data with a maximum occurrence of {max_occur} patches per angle. Plotting results...")   
        
        for i in range(len(list_dict)):
            label = labels[i]                           # Get the label for the current patch
            mask = labels[i] != -1                         # Create a mask to ignore invalid pixels
            mean_angle = int(np.floor(np.mean(label[mask]) % 180))  # Calculate the mean angle in the range [0,180]

            if occurrences.get(mean_angle, 0) > max_occur:  # If the mean angle exceeds the maximum allowed occurrence
                occurrences[mean_angle] -= 1     # Decrease the occurrence count
                disc_index.append(i)         # Track this patch index for discarding

        for i, (patch, label, list_dict) in enumerate(zip(patches, labels, list_dict)):
            if i not in disc_index:                         # If the patch index is not marked for discarding
                data.append((patch, label, list_dict))        # Add the patch, label, and metadata to the balanced dataset
                
        plot_occurrences(occurrences, max_occur, threshold, 'after_balancing', self.plot_path)  # Plot occurrences after balancing
        return data  # Return the balanced dataset
    
    
    ### Creating datasets by noise adiciton ####
    
    def _apply_brightness_contrast(self, data_tuple):
        """
        Apply brightness and contrast adjustments to the dataset according to brightness and contrast factor
        """
        
        patches, labels, list_dict = self._shuffle_data(data_tuple)  # Shuffle the data
        for i in range(len(patches)):
            num_slices, height, width = patches[i].shape     # Get the dimensions of the current patch
            adjusted_slices = []                            # Initialize a list to store adjusted image slices

            for slice_idx in range(num_slices):
                image_slice = patches[i][slice_idx]                            # Get the current image slice
                image = Image.fromarray((image_slice * 255).astype(np.uint8))       # Convert to PIL image for processing
                image = ImageEnhance.Brightness(image).enhance(self.bright_factor)      # Apply brightness adjustment
                image = ImageEnhance.Contrast(image).enhance(self.cont_factor)          # Apply contrast adjustment
                adjusted_slice = np.array(image).astype(np.float32) / 255          # Convert back to numpy array and normalize
                adjusted_slices.append(adjusted_slice)                        # Add adjusted slice to the list

            patches[i] = np.stack(adjusted_slices)      # Stack the adjusted slices into the final patch
        return patches, labels, list_dict           # Return the adjusted patches, labels, and metadata
   
    def _apply_gaussian_noise(self, data_tuple):
        """
        Apply Gaussian noise to the dataset according to g_mean and g_std parameters
        """
        
        patches, labels, list_dict = self._shuffle_data(data_tuple)  # Shuffle the data
        for i in range(len(patches)):
            num_slices, height, width = patches[i].shape                        # Get the dimensions of the current patch
            noise = np.random.normal(self.g_mean, self.g_std, (num_slices, height, width))  # Generate Gaussian noise
            patches[i] = np.clip(patches[i] + noise, 0, 1)                               # Add noise to the patch and clip to valid range
        return patches, labels, list_dict                           # Return the noisy patches, labels, and metadata

    def _apply_speckle_noise(self, data_tuple):
        """
        Apply speckle noise to the dataset according to s_mean and s_std parameters
        """
        
        patches, labels, list_dict = self._shuffle_data(data_tuple)  # Shuffle the data
        for i in range(len(patches)):
            num_slices, height, width = patches[i].shape                       # Get the dimensions of the current patch
            noise = np.random.normal(self.s_mean, self.s_std, (num_slices, height, width))    # Generate speckle noise
            patches[i] = np.clip(patches[i] + patches[i] * noise, 0, 1)                # Add speckle noise and clip to valid range
        return patches, labels, list_dict                                    # Return the noisy patches, labels, and metadata
    
    def _apply_custom_noise(self, data_tuple):
        """
        Apply custom noise to a random selection of slices in the dataset.
        This noise was tailored for strange random peaks in ambiguous line profiles.
        Defines a random number of points (from the 24 of the line profiles) to add noise.
        Defines a random value of noise (from a defined random distribution) to add to each selected point.
        """
        
        patches, labels, list_dict = self._shuffle_data(data_tuple)  # Shuffle the data
        for i in range(len(patches)):
            num_slices = np.random.randint(1, self.max_slices + 1)                    # Determine the number of slices to add noise to
            slice_indices = np.random.choice(patches[i].shape[0], num_slices, replace=False)  # Randomly select slices to add noise
            
            for slice_idx in slice_indices:
                noise_factor = np.random.uniform(self.lower_value, self.higher_value)  # Determine noise factor
                patches[i][slice_idx] = np.clip(patches[i][slice_idx] + noise_factor, 0, 1)  # Apply noise and clip values

        return patches, labels, list_dict  # Return the noisy patches, labels, and metadata
    
    def _normalize_patch(self, image):
        """
        Normalize patch to the range [0, 1]
        """
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def _standardize_patch(self, image):
        """
        Standardize patch to have a mean of 0 and a standard deviation of 1
        """
        return (image - np.mean(image)) / np.std(image)
    
    def _update_train_prof_samples(self, cv_sets, train_prof_samples):
        """
        Update the training data and labels samples selected earlier in the patch processing phase (set_plot_entries).
        Plot those training profile samples for each dataset to monitor line profiles and patch location.
        """
        
        datasets = ['Augmented', 'Augmented + Contrast Adjust', 'Augmented + Gaussian Noise', 'Augmented + Speckle Noise', 'Augmented + Custom Noise']
        for fold_index, fold in enumerate(cv_sets):
            self.logger.info(f"Plotting fold {fold_index} data to monitor line profiles...")
            
            new_prof_samples = []  # Initialize a list to store new profile samples

            for prof_image, prof_label, prof_dict in train_prof_samples:
                for train_image, train_label, train_dict in fold:
                    if train_dict['image'] == prof_dict['image'] and train_dict['location'] == prof_dict['location']:
                        new_prof_samples.append((train_image, train_label, train_dict))
                        break  # Exit loop once a match is found
            
            # Plot data for the updated samples
            plot_line_profiles(new_prof_samples, os.path.join(self.plot_path, 'check_line_profiles/', 'train/'), paths['data'], phase=f'{datasets[fold_index]} dataset ({fold_index})')
    
    def create_cv_sets(self, dataset, train_prof_samples):
        """
         Create cross-validation sets by splitting the augmented dataset into folds and applying different noise addiction
        """
        
        kf = KFold(n_splits=self.k_folds, shuffle=True)  # Initialize KFold cross-validation with shuffling
        cv_sets = []                                   # Initialize a list to store the cross-validation sets
        self.logger.info(f"Iniciating cross-validation dataset creation with {self.k_folds} folds...")

        for fold, (train_idx, _) in enumerate(kf.split(dataset)):
            self.logger.info(f"Creating fold {fold} dataset.")
            train_data = [dataset[i] for i in train_idx]  # Select the training data for the current fold
            
            # Normalize image data per patch (firstly pixel normalization was selected but this gives better learning)
            for i in range(len(train_data)):
                patch, label, data_dict = train_data[i]
                train_data[i] = (self._normalize_patch(patch), label, data_dict)
                #train_data[i] = (self._standardize_patch(patch), label, data_dict)          # uncomment the following line to standardize instead of normalize
                # Note: This normalization step will also be applied to test and validation data

            if fold == 0:
                # Augmented dataset
                cv_sets.append(train_data)
            elif fold == 1:
                # Brightness/Contrast Adjustment
                patches, labels, dicts = self._apply_brightness_contrast(train_data)
                cv_sets.append(list(zip(patches, labels, dicts)))
            elif fold == 2:
                # Gaussian Noise
                patches, labels, dicts = self._apply_gaussian_noise(train_data)
                cv_sets.append(list(zip(patches, labels, dicts)))
            elif fold == 3:
                # Speckle Noise
                patches, labels, dicts = self._apply_speckle_noise(train_data)
                cv_sets.append(list(zip(patches, labels, dicts)))
            elif fold == 4:
                # Custom Noise
                patches, labels, dicts = self._apply_custom_noise(train_data)
                cv_sets.append(list(zip(patches, labels, dicts)))

        self.logger.info(f"Datasets creation complete.")
        
        # Update train profile samples only if they are provided
        if train_prof_samples is not None:
            self._update_train_prof_samples(cv_sets, train_prof_samples)  # Monitor line profiles across CV sets

        return cv_sets  # Return the created cross-validation sets