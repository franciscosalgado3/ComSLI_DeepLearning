# Master Thesis Project: Deep Learning Workflow for ComSLI Fiber Reconstruction

This repository contains a deep learning framework for predicting unidirectional orientations of Computational Scattered Light Imaging (ComSLI) fiber data following Francisco Salgado's Master thesis project. This project was conducted in the Menzel Lab (https://menzellab.gitlab.io/) at the Imaging Physics Department of the Delft University of Technology (TU Delft) under the supervision of Dr. Miriam Menzel (TU Delft) and Dr. Hélder Oliveira (FCUP). 

In this study, Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN) and Generative Adversarial Networks (GAN) models were trained and tested on ComSLI high micro-resolution artificial brain samples. This project involves deep learning regression tasks, utilizing supervised and semi/self-supervised methodologies. It incorporates hybrid configurations of LSTM and CNN models to process image data and integrates a GAN model inspired by the GAIN framework. The project is structured to handle data preprocessing, dataset creation, model training, evaluation, and result visualization. Preprocessing methods, including patch-processing, data augmentation, and noise addition, were critical in producing accurate learning Fiber Orientation Maps (FOMs) and addressing missing or incorrect labeling by Scattered Ligth Imaging Toolbox (SLIX, https://github.com/3d-pli/SLIX#2-run-slix).


### Pixel Regression/Semantic Segmentation Models:
The LSTM-CNN models demonstrated consistent reliability, with the LSTM + 3D U-Net model achieving a total average angular error of approximately 5º. Despite challenges in generalization, the GAN model provided higher resolution FOMs in well-labeled regions- it showed an average angular error of 3º- highlighting its potential with further refinements.

- **LSTM + 2D/3D CNN/U-Net**: Predicts the labeled pixel orientations in supervised learning, extrapolating labels in the unlabeled pixels.
- **GAN (CNN-based)**: Predicts the labeled pixels in supervised learning and inputs orientations non-labeled pixel in a self/semi-supervised learning.


## Project Structure
```
├── models/
│ ├── model_lstm.py # LSTM model implementation (obsolote model)
│ ├── hybrid_lstm_cnn.py # Hybrid LSTM-CNN model implementation (obsolete models)
│ ├── model_lstm_cnn.py # LSTM-CNN (2D CNN and 3D CNN) model implementation
│ ├── gan_model.py # GAN moodel implementation
├── init.py # Initialization file for the model package
├── config.py # Configuration settings for file paths and directories
├── dataset.py # Dataset handling and processing (including pipeline for dataset creation)
├── early_stop.py # Early stopping mechanism for training
├── loggs.py # Centralized logging setup
├── pre_processing.py # Data preprocessing steps and augmentation
├── testing_dp.py # Testing data pipeline and evaluation for LSTM-CNN models
├── training_dp.py # Training data pipeline and model training for LSTM-CNN models
├── testing_gan.py # Testing data pipeline and evaluation for GAN model
├── training_gan.py # Training data pipeline and model training for the GAN model
├── visualization.py # Visualization utilities for data and results
```

## Overview

This project is designed to facilitate the development, training, and evaluation of LSTM, CNN, and GAN models for regression tasks. The key components of the project include:

- **Models**: Implementations of LSTM, LSTM-CNN, and GAN models for handling ComSLI image data.
- **Preprocessing**: Data augmentation and preprocessing steps required to prepare the datasets for training and testing.
- **Dataset**: Pipeline that ensures the creation of training sets regarding different noise addiction to evaluate their performance on also created validation and testing datasets.
- **Training**: A structured pipeline for training the models, including loss monitoring, early stopping, and logging.
- **Testing**: Evaluation scripts to assess model performance using various metrics.
- **Visualization**: Tools to visualize data, model 1D FOM predictions, and training results.


## Future Development

This project's goal was to build a complete framework following a "Multi-tasking model" approach agregating segmebntation and a regression task. In a first stage a segmentation model aims to identify background, 1-fiber, and 2-fiber regions and the regression model precidting the acutal orientation in the pixel.
The objective of this project was to develop a framework utilizing a multi-tasking model approach that integrates both segmentation and regression tasks. In the initial phase, a segmentation model is employed to accurately identify background, single-fiber, and dual-fiber regions within ComSLI data. Subsequently, a regression model among the referred ones is implemented to predict the orientation of each pixel. However, the full implementation was not feasible as the segmentation model remains was not done. Future development will focus on developing and integrating this segmentation component to fully realize the framework's potential.  

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
- Imageio

You can install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Data Requirements

This project operates on specific data formats and conventions:

- **Image Files**: The input images should be the ComSLI NIfTI stack files (.nii format).
- **Label Files**: The labels must be provided as direction_1.tif files, which indicate the orientation of the fibers. These label files must be masked in the background and in regions where there are crossing fibers (denoted as 2-fiber regions). Only regions with a single fiber should be unmasked, ensuring that the model focuses on the relevant fiber orientations.

### Dataset Creation and Data Preprocessing

The first step in using this framework is to prepare the datasets. This is done by running `dataset.py`, which handles the entire data preparation pipeline:

1. **Data Loading**: `dataset.py` fetches the raw data (not included in this repository) and initiates the preprocessing steps.
2. **Patch Processing**: The script calls functions from `pre_processing.py` to handle patch processing, including normalization, augmentation, and extraction of patches from the original dataset.
3. **Data Augmentation**: The `Augment_Dataset` class functions are utilized to augment the data. This step enhances the dataset by applying various transformations to increase its diversity.
4. **Dataset Creation**: The processed and augmented data is then used to create training, validation, and testing datasets. This is done by calling the relevant functions from the `DatasetCreator` class, which is also defined in `dataset.py`.

**Note**: Ensure that the dataset is correctly prepared by checking the outputs in the specified directories.

### Training

Once the datasets are prepared, you can proceed to train your models:

- **Run Training**: To train the models, use the training pipeline provided in `training_dp.py` or `training_gan.py`. This script loads the training and validation datasets created in `dataset.py`, and uses models defined in the `models/` directory.
- **Model Training**: The `Trainer` class in `training_dp.py` (and `training_gan.py`) orchestrates the training process, handling the training loop. It also applies the early stopping mechanism from `early_stop.py` to automatically prevent overfitting.
- **Visualization**: During and after training, the script produces various plots to visualize the training progress, losses, and other metrics. These plots are generated using functions from `visualization.py`.

### Testing and Evaluation

After training, you can evaluate your models by running the `testing_dp.py` (or `testing_gan.py`) script:

- **Load Testing Data**: The testing script loads the test datasets prepared by `dataset.py`.
- **Model Evaluation**: The `Tester` class in `testing_dp.py` (or `testing_gan.py`) evaluates the model's performance on the test data. It uses the same models trained earlier and produces performance metrics such as loss and R-squared values.
- **Result Visualization**: The evaluation results, including prediction accuracy and comparison between predicted and actual values, are visualized using functions from `visualization.py`.

### Logging

The project uses a centralized logging system defined in `loggs.py`. All significant steps in the training, testing, and preprocessing pipelines are logged for easier debugging and tracking of progress.

### Early Stopping

The `early_stop.py` script is used during training to monitor the model's performance on the validation set. If the performance stops improving, training is halted early to prevent overfitting and unnecessary computation.

- Francisco Salgado
