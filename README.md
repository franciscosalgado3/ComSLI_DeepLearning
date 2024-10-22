# Master Thesis Project

This repository contains a deep learning workflow designed for nevre fiber orientation prediction through Computational Scattered Light Imaging data. It consists of an image learning regression task at the pixel-level, leveraging LSTM-CNN models in hybrid configurations, including a GAN approach to input orientations in non-labeled pixels. The repository is structured to handle data preprocessing, dataset creation, model training, evaluation, and results visualization.
This project was conducted in Delft University of Technology (TU Delft) in the Menzel Lab (Imaging Physics Department) under the supervision of Dr. Miriam Menzel (TU Delft) and  Dr. Hélder Oliveira (FCUP) 

### Segmentation Model:
- **LSTM + 3D CNN**: Aims to identify background, 1-fiber, and 2-fiber regions.

### Regression Models:
- **LSTM + 2D/3D CNN**: Predicts the labeled pixel orientations.
- **GAN (CNN-based)**: Predicts the non-labeled pixel orientations (label imputation).

The GAN and segmentation models, which will be integrated into the project later, share key components like dataset handling, early stopping, and visualization. However, the segmentation model employs a different preprocessing approach tailored to its specific needs. Each of these models has uniquely designed training and testing loops to address their particular challenges.

This repository currently represents the core of the regression framework (**LSTM + 2D/3D CNN**).

## Project Structure
```
├── models/
│ ├── hybrid_lstm_cnn.py # Hybrid LSTM-CNN model implementation
│ ├── model_lstm_cnn.py # Phased LSTM-CNN (2D CNN and 3D CNN) model implementation
│ ├── model_lstm.py # LSTM model implementation
├── init.py # Initialization file for the models package
├── config.py # Configuration settings for file paths and directories
├── dataset.py # Dataset handling and processing (including pipeline for dataset creation)
├── early_stop.py # Early stopping mechanism for training
├── loggs.py # Centralized logging setup
├── pre_processing.py # Data preprocessing steps and augmentation
├── testing_dp.py # Testing data pipeline and evaluation
├── training_dp.py # Training data pipeline and model training
├── visualization.py # Visualization utilities for data and results
```

## Overview

This project is designed to facilitate the development, training, and evaluation of LSTM and CNN-based models for regression tasks. The key components of the project include:

- **Models**: Implementations of LSTM, CNN, and hybrid LSTM-CNN models for handling ComSLI image data.
- **Preprocessing**: Data augmentation and preprocessing steps required to prepare the datasets for training and testing.
- **Dataset**: Pipeline that ensures the creation of training sets regarding different noise addiction to evaluate their performance on also created validation and testing datasets.
- **Training**: A structured pipeline for training the models, including loss monitoring, early stopping, and logging.
- **Testing**: Evaluation scripts to assess model performance using various metrics.
- **Visualization**: Tools to visualize data, model 1D FOM predictions, and training results.


## Setup

### Prerequisites

Ensure that you have the following dependencies installed:

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- nibabel
- PIL

You can install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Data Requirements

This project operates on specific data formats and conventions:

- **Image Files**: The input images should be the ComSLI NIfTI stack files (.nii format).
- **Label Files**: The labels must be provided as direction_1.tif files, which indicate the orientation of the fibers. It is crucial that these label files are masked in the background and in regions where there are crossing fibers (denoted as 2-fiber regions). Only regions with a single fiber should be unmasked, ensuring that the model focuses on the relevant fiber orientations.

### Dataset Creation and Data Preprocessing

The first step in using this framework is to prepare the datasets. This is done by running `dataset.py`, which handles the entire data preparation pipeline:

1. **Data Loading**: `dataset.py` fetches the raw data (not included in this repository) and initiates the preprocessing steps.
2. **Patch Processing**: The script calls functions from `pre_processing.py` to handle patch processing, including normalization, augmentation, and extraction of patches from the original dataset.
3. **Data Augmentation**: The `Augment_Dataset` class functions are utilized to augment the data. This step enhances the dataset by applying various transformations to increase its diversity.
4. **Dataset Creation**: The processed and augmented data is then used to create training, validation, and testing datasets. This is done by calling the relevant functions from the `DatasetCreator` class, which is also defined in `dataset.py`.

**Note**: Ensure that the dataset is correctly prepared by checking the outputs in the specified directories.

### Training

Once the datasets are prepared, you can proceed to train your models:

- **Run Training**: To train the models, use the training pipeline provided in `training_dp.py`. This script loads the training and validation datasets created in `dataset.py`, and uses models defined in the `models/` directory.
- **Model Training**: The `Trainer` class in `training_dp.py` orchestrates the training process, handling the training loop. It also applies the early stopping mechanism from `early_stop.py` to automatically prevent overfitting.
- **Visualization**: During and after training, the script produces various plots to visualize the training progress, losses, and other metrics. These plots are generated using functions from `visualization.py`.

### Testing and Evaluation

After training, you can evaluate your models by running the `testing_dp.py` script:

- **Load Testing Data**: The testing script loads the test datasets prepared by `dataset.py`.
- **Model Evaluation**: The `Tester` class in `testing_dp.py` evaluates the model's performance on the test data. It uses the same models trained earlier and produces performance metrics such as loss and R-squared values.
- **Result Visualization**: The evaluation results, including prediction accuracy and comparison between predicted and actual values, are visualized using functions from `visualization.py`.

### Logging

The project uses a centralized logging system defined in `loggs.py`. All significant steps in the training, testing, and preprocessing pipelines are logged for easier debugging and tracking of progress.

### Early Stopping

The `early_stop.py` script is used during training to monitor the model's performance on the validation set. If the performance stops improving, training is halted early to prevent overfitting and unnecessary computation.



- Francisco Salgado
