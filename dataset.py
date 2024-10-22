import os                                                        # Importing the os module to interact with the operating system's file system
import random                                                    # Importing random for random number generation
import torch                                                     # Importing PyTorch for working with datasets
import numpy as np                                               # Importing numpy for numerical operations
from loggs import logger                                         # Importing the centralized logger
from torch.utils.data import Dataset                             # Importing Dataset class for PyTorch datasets

from config import paths                                                         # Import paths from the configuration file
from visualization import plot_occurrences, plot_line_profiles                   # Importing plotting functions
from pre_processing import Load_n_Normalize, Patch_Processor, Augment_Dataset    # Import preprocessing classes

class DatasetCreator(Dataset):
    """
    This class is responsible for the creation, management, and saving of datasets for training, validation, and testing. 
    It allows the dataset to be processed, shuffled, saved, and retrieved in different subsets (training, validation, testing).
    """

    def __init__(self, data=None, preloaded_data=None, train_size=None, test_size=None, val_size=None):
        """
        Initialize the DatasetCreator with optional preloaded data and specific sizes for subsets
        """
        
        self.logger = logger                                             # Logger instance for logging messages
        self.data = {'training': [], 'validation': [], 'testing': []}    # Dictionary to store data subsets
        self.current_data = None                                         # Placeholder for the current dataset being worked on
        self.current_subset = None                                       # Placeholder for the current subset being worked on
        self.val_size = val_size                                         # Size of the validation set
        self.test_size = test_size                                       # Size of the test set
        self.train_size = train_size                                     # Size of the training set

        if preloaded_data:               # If preloaded data is provided, the class skips the data processing steps and directly uses the provided data.
            self.data = preloaded_data       #This is useful if data is already processed and ready to be used.
        elif data is not None:
            self._process_data(data)         # Process and add the data
            self._limit_subset_size()        # Limit the size of the subsets if specified


    def _process_data(self, data):
        """
        Process the provided data and add it to the appropriate subset
        """
        
        for image, label, data_dict in data:
            self._add_data(image, label, data_dict)  # Add each data entry to the appropriate subset

    def _add_data(self, image, label, data_dict):
        """
        Add a single data entry to the appropriate subset (training, validation, testing)
        """
        
        subset = data_dict.get('type', 'training')  # Determine which subset the data belongs to
        if subset not in self.data:
            raise ValueError(f"Invalid subset type provided: {subset}")  # Raise an error if the subset type is invalid

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # Convert image to a PyTorch tensor and reorder dimensions to (detph=24, heigth, width)
        label_tensor = torch.from_numpy(label).float()              # Convert label to a PyTorch tensor
        
        self.data[subset].append((image_tensor, label_tensor, data_dict))  # Add the tensors and metadata to the appropriate subset

    def _limit_subset_size(self):
        """
        Limit the size of the subsets (training, validation, testing) if sizes are specified
        """
        
        if self.train_size is not None:
            self._limit_subset('training', 1 - self.train_size)
        if self.val_size is not None:
            self.create_validation_set(self.val_size)
        if self.test_size is not None:
            self._limit_subset('testing', self.test_size)

    def _limit_subset(self, subset_name, subset_ratio):
        """
        Limit the size of a specific subset based on the provided ratio.
        """
        
        if subset_name in self.data and self.data[subset_name]:
            total_training_count = len(self.data['training'])  # Get the total number of training samples
            max_count = int(total_training_count * subset_ratio)  # Calculate the maximum number of samples allowed
            current_count = len(self.data[subset_name])  # Get the current number of samples in the subset
            
            if subset_name == 'training':
                angle_count = int((total_training_count - max_count) / 180)  # Calculate the number of samples per angle
                new_set = []
                for i in range(180):
                    angle_data = [item for item in self.data['training'] if item[2].get('mean_angle_patch') == i]  # Filter data by angle
                    keep_data = np.random.choice(len(angle_data), angle_count, replace=False)  # Randomly select samples
                    new_set.extend([angle_data[idx] for idx in keep_data])  # Add selected samples to the new set
                self.data['training'] = self._shuffle_data(new_set)  # Shuffle the new set and update the training data
                logger.info(f"Limited {subset_name} size to {max_count} samples")  # Log the limiting action

            elif subset_name == 'testing':
                max_count = min(max_count, current_count)  # Ensure the max count doesn't exceed the current count
                indices = np.random.choice(current_count, max_count, replace=False)  # Randomly select samples
                self.data['testing'] = [self.data[subset_name][i] for i in indices]  # Update the testing data with selected samples
                logger.info(f"Limited {subset_name} size to {max_count} samples")  # Log the limiting action
    
    def _shuffle_data(self, data):
        """
        Shuffle the data to randomize the order
        """
        
        indices = np.arange(len(data))    # Create an array of indices
        np.random.shuffle(indices)         # Shuffle the indices
        return [data[i] for i in indices]    # Reorder the data according to the shuffled indices

    def save(self, path):
        """
        Save the entire dataset to the specified path
        """
        
        logger.info(f"Saving dataset to {path}...")  # Log the action
        with open(path, 'wb') as f:
            torch.save(self.data, f)  # Save the dataset using PyTorch's save function

    @classmethod
    def save_set(cls, data, path, train_size=None, test_size=None, val_size=None):
        """
        Class method to save a dataset to the specified path
        """
        
        instance = cls(data=data, train_size=train_size, val_size=val_size, test_size=test_size)  # Create an instance with the provided data
        instance.save(path)  # Save the dataset

    def save_cv_sets(self, cv_sets, save_path):
        """
        Save cross-validation sets to the specified path
        """
        
        save_cv_path = os.path.join(save_path, 'datasets')  # Create the save path
        os.makedirs(save_cv_path, exist_ok=True)  # Ensure the directory exists
        
        for fold, train_data in enumerate(cv_sets):
            train_tensors = [(torch.from_numpy(p).float().permute(2, 0, 1), torch.from_numpy(l).float(), d) for p, l, d in train_data]  # Convert data to PyTorch tensors
            
            train_save_path = os.path.join(save_cv_path, f'train_set_{fold}.pth')  # Create the file path for each fold
            torch.save(train_tensors, train_save_path)  # Save the tensors

        logger.info(f"Saved all datasets at {save_cv_path}")  # Log the saving action
        
        
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

    def create_validation_set(self, test_data, val_size_ratio):
        """
        Create a validation set from the test data based on a specified ratio
        """
        
        # Normalize image data per pixel initially
        for i in range(len(test_data)):
            patch, label, data_dict = test_data[i]
            test_data[i] = (self._normalize_patch(patch), label, data_dict)
            # Uncomment the following line to standardize instead of normalize
            #test_data[i] = (self._standardize_patch(patch), label, data_dict)
        
        total_test_count = len(test_data)  # Total number of test samples
        val_count = int(total_test_count * val_size_ratio)  # Calculate the number of validation samples

        # Randomly select indices for the validation set
        selected_indices = np.random.choice(total_test_count, val_count, replace=False)
        
        # Create validation set and update test set
        val_set = []
        test_set = []
        for i, item in enumerate(test_data):
            if i in selected_indices:
                patch, label, data_dict = item
                data_dict['type'] = 'validation'
                val_set.append((patch, label, data_dict))
            else:
                test_set.append(item)
        
        # Store the sets
        self.data['validation'] = self._shuffle_data(val_set)
        self.data['testing'] = test_set  # No need to shuffle as it's already in random order
        self.logger.info(f"Created validation set with {val_count} samples from testing patches")  # Log the creation of the validation set
        
        return val_set, test_set  # Return the validation and test sets

    def report_dataset_info(self):
        """
        Report information about the dataset, including counts of different types of data
        """
        
        def count_types(data, type_filter):
            counts = {'augmented': 0, 'modified': 0, 'raw': 0, 'not modified': 0}
            for _, _, data_dict in data:
                if data_dict['type'] == type_filter:
                    if 'nature' in data_dict:
                        nature = data_dict['nature']
                        counts[nature] += 1
            return counts  # Return counts of different types of data

        train_counts = count_types(self.data['training'], 'training')
        validation_counts = count_types(self.data['validation'], 'validation')
        test_counts = count_types(self.data['testing'], 'testing')

        # Log the summary of the dataset
        logger.info("Dataset Information Summary:")
        logger.info(f"- Training Data (Fold 0): {len(self.data['training'])} patches  |  Augmented: {train_counts['augmented']}, Raw: {train_counts['raw']}, Modified: {train_counts['modified']}, Not modified: {train_counts['not modified']}")
        logger.info(f"- Validation Data: {len(self.data['validation'])} patches | Augmented: {validation_counts['augmented']}, Raw: {validation_counts['raw']}, Modified: {validation_counts['modified']}, Not modified: {validation_counts['not modified']}")
        logger.info(f"- Testing Data: {len(self.data['testing'])} patches    |  Augmented: {test_counts['augmented']}, Raw: {test_counts['raw']}, Modified: {test_counts['modified']}, Not modified: {test_counts['not modified']}")
    
    
    def __len__(self):
        """
        Return the length of the current dataset
        """
        return len(self.current_data)

    def __getitem__(self, idx):
        """
        Return a specific item from the current dataset based on the index
        """
        return self.current_data[idx]

    def get_subset(self):
        """
        Return the name of the current subset
        """
        return self.current_subset
        
        
class DatasetProcessingPipeline:
    """
    This class dfeines the pipeline for the datasets creation for training, testing and validation
    """
    
    def __init__(self, train_image_files, train_label_files, test_image_files, test_label_files, logger,
                patch_size, allow_modification, neighbors, stride, plot_dir, num_prof_plots, augment_ratio, k_folds,
                val_size_ratio, preloaded_data, train_size, test_size, val_size,
                g_mean, g_std, s_mean, s_std, cont_factor, bright_factor, max_slices, lower_value, higher_value):
        
        self.train_image_files = train_image_files     # List of file paths for training images
        self.train_label_files = train_label_files     # List of file paths for training labels
        self.test_image_files = test_image_files       # List of file paths for testing images
        self.test_label_files = test_label_files       # List of file paths for testing labels
        
        self.patch_size = patch_size                    # Size of patches to be extracted from the images
        self.allow_modification = allow_modification    # Whether to allow patch modification (e.g., augmenting)
        self.neighbors = neighbors                      # Number of neighbors to consider during patch processing
        self.stride = stride                            # Stride size for patch extraction
        self.plot_dir = plot_dir                        # Directory where plots will be displayed
        self.num_prof_plots = num_prof_plots            # Number of monitoring patches to generate plots
        self.logger = logger                            # Logger instance for logging pipeline progress
        self.augment_ratio = augment_ratio              # Ratio of augmentation patches to total patches
        self.k_folds = k_folds                          # Number of cross-validation folds datasets
        
        self.val_size_ratio = val_size_ratio      # Ratio of validation set according to testing size (if not preloaded_data)
        self.preloaded_data = preloaded_data      # Preloaded data (if provided)
        self.train_size = train_size              # Size of the training set (if preloaded data is provided)
        self.test_size = test_size                # Size of the test set (if preloaded data is provided)
        self.val_size = val_size                  # Size of the validation set (if preloaded data is provided)
        
        self.g_mean = g_mean                    # Mean for Gaussian noise
        self.g_std = g_std                      # Standard deviation for Gaussian noise
        self.s_mean = s_mean                    # Mean for speckle noise
        self.s_std = s_std                      # Standard deviation for speckle noise
        self.bright_factor = bright_factor      # Factor for adjusting brightness
        self.cont_factor = cont_factor          # Factor for adjusting contrast
        self.max_slices = max_slices            # Maximum number of slices to apply custom noise
        self.higher_value = higher_value        # Upper bound for custom noise
        self.lower_value = lower_value          # Lower bound for custom noise
        
        
    def run(self):
        """
        If preloaded data is provided, skip data loading, preprocessing, and augmentation.
        otherwise Dataseprocessing pipeline is runned normally (default)
        """
        
        if self.preloaded_data:
            self.logger.info("Using preloaded data. Skipping data loading, preprocessing, and augmentation.")
            final_train_data = self.preloaded_data['training']
            test_processed_data = self.preloaded_data['testing']
            train_prof_samples = test_prof_samples = None  # Assuming these aren't needed if preloaded_data is given

        else:
            # Load and preprocess the data
            train_loaded_data, test_loaded_data = Load_n_Normalize(
                self.train_image_files, self.train_label_files, self.test_image_files, self.test_label_files, self.plot_dir
            ).preprocess()  # Load and normalize the images and labels
            
            train_processed_data, test_processed_data, train_prof_samples, test_prof_samples = Patch_Processor(
                self.patch_size, self.allow_modification, self.neighbors, self.plot_dir, self.num_prof_plots, self.stride
            ).process_dataset(train_loaded_data, test_loaded_data)
            
            
            # Monitor the line profiles of the patches (for quality control or analysis)
            self.logger.info(f"Plotting train and test data to monitor line profiles..")   #this takes some minutes
            plot_line_profiles(
                train_prof_samples, os.path.join(self.plot_dir, 'check_line_profiles/', 'train/'), paths['data'], phase='Raw patches'
            )
            plot_line_profiles(
                test_prof_samples, os.path.join(self.plot_dir, 'check_line_profiles/', 'test/'), paths['data'], phase='Raw patches'
            )
            
            # Augment and balance the training data
            self.logger.info(f"Starting data augmentation process and cross-validation datasets..")
            augment_dataset = Augment_Dataset(
                self.augment_ratio, self.plot_dir, self.k_folds, self.g_mean, self.g_std, self.s_mean, self.s_std, self.cont_factor, self.bright_factor, self.max_slices, self.higher_value, self.lower_value
            )
            final_train_data = augment_dataset.balance_train_data(train_processed_data)    # Augment and balance the training data
        
        # Create cross-validation sets, include monitoring line profiles for each dataset fold
        cv_sets = augment_dataset.create_cv_sets(final_train_data, train_prof_samples)
        
        # Create DatasetCreator instance and process cross-validation sets
        self.logger.info(f"Saving all training, validation and tetsing datasets..")
        dataset = DatasetCreator(data=final_train_data, preloaded_data=self.preloaded_data, train_size=self.train_size, test_size=self.test_size, val_size=self.val_size)
        
        if cv_sets:
            dataset.save_cv_sets(cv_sets, self.plot_dir)   # Save the cross-validation sets if created
                
        # Create and save the validation set and test sets
        val_data, test_data = dataset.create_validation_set(test_processed_data, self.val_size_ratio)
        dataset.save_set(val_data, os.path.join(self.plot_dir, 'datasets/', 'val_set.pth'))
        dataset.save_set(test_data, os.path.join(self.plot_dir, 'datasets/', 'test_set.pth'))

        # Display dataset information summary
        dataset.report_dataset_info()
        
if __name__ == '__main__':
    
    #lstm-cnn
    train_images = ['105_stack.nii', '158_stack.nii']        # train images names
    train_labels = ['105_dir_1.tif', '158_dir_1.tif']        # train labels names
    test_images = ['106_stack.nii']                          # test images names
    test_labels = ['106_dir_1.tif']                          # test labels names

    train_image_files = [os.path.join(paths['data'], file_name) for file_name in train_images]        # array with train images paths
    train_label_files = [os.path.join(paths['data'], file_name) for file_name in train_labels]        # array with train labels paths
    test_image_files = [os.path.join(paths['data'], file_name) for file_name in test_images]         # array with test image paths
    test_label_files = [os.path.join(paths['data'], file_name) for file_name in test_labels]         # array with test label paths
    
    plot_dir = os.path.join(paths['preprocessing'])  # base directory of all preprocessing plots
    os.makedirs(plot_dir, exist_ok=True)

    #### Pre-processing
    patch_size = 16               # size of the patch
    allow_modification = True     # allows neighbouring inputing
    neighbors = 5                 # set the number of valid neighbour 
    stride = 0                    # stride applied to the patch processing (0 = no stride)
    num_prof_plots = 1            # number of different patches to plot per case (changing this one needs changing the code in update_line_profiles taking out the break condition)

    #### Data Augmentation
    augment_ratio = 0.8        # ratio of augmented patches to original images
    k_folds=5                  # number of cross-validation folds         
    
    # Brightness/Constrast Adjustment
    bright_factor = 1.2       # brightness factor
    cont_factor = 1.5         # contrast factor
    
    # Gaussian Noise
    g_mean = 0.5
    g_std = 0.2
    
    # Speckle Noise
    s_mean = 0.8
    s_std = 0.5
    
    # Custom Noise (fine-tune this values)
    max_slices = 15         # max number of slices where noise would be randomly applied (in this case 15/24)
    lower_value = -0.2       # lower boundary for random noise adition
    higher_value = 0.6        # higher boundary for random noise adition
    
    ### Datasets
    val_size_ratio = 0.25        # Ratio of validation set from testing data (if preloaded_data not given)
    
    preloaded_data = None       # Structure: { 'training': [(image_tensor1, label_tensor1, data_dict1),...],
                                #              'validation': [(image_tensor2, label_tensor2, data_dict2),..],
                                #              'testing': [(image_tensor3, label_tensor3, data_dict3),...] }
    
    train_size = None       # set size if needded and only if preloaded_data is given 
    test_size = None        # set size if needded and only if preloaded_data is given
    val_size = None        # set size if needded and only if preloaded_data is given
    
    # Run the pipelin
    DatasetProcessingPipeline(
        train_image_files, train_label_files, test_image_files, test_label_files, logger, 
        patch_size, allow_modification, neighbors, stride, plot_dir, num_prof_plots, augment_ratio, k_folds,
        val_size_ratio, preloaded_data, train_size, test_size, val_size,
        g_mean, g_std, s_mean, s_std, cont_factor, bright_factor, max_slices, lower_value, higher_value
    ).run()
    