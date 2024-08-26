# Lung Cancer Classification using VGG19 Transfer Learning

## Overview

This project involves classifying lung cancer images using a VGG19 model with transfer learning. The dataset has been enhanced using ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) for improved image quality. The enhanced images are then processed and used to train a VGG19 model to classify lung cancer types.

## Dataset

The dataset is stored in the directory `D:/pythonProject/lung cancer GAN/Data`, organized into the following structure:

- `train/`
  - `remaining/`
    - Contains images to be used for training.

The images are preprocessed to 128x128 pixels, resized with padding to maintain aspect ratio, and normalized.

## Preprocessing

The following steps are performed on the dataset:

1. **Normalization**: Images are normalized to the range [0, 1].
2. **Resizing with Padding**: Images are resized to 128x128 pixels while maintaining aspect ratio using padding.

## Training

A VGG19 model is used for classification with the following modifications:

- The model is loaded with pre-trained weights.
- The final fully connected layer is replaced to match the number of classes in the dataset.
- Training is performed using the modified classifier layer, with weights of the other layers frozen.

### Training Details

- **Epochs**: 25
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: SGD with momentum
- **Scheduler**: StepLR with a step size of 7 and gamma of 0.1

## Results

The VGG19 model achieved an accuracy of around **99%** on the validation set. The following plots are generated during training:

- **Loss vs. Epochs**
- **Accuracy vs. Epochs**

## Code

The complete code for preprocessing, training, and saving the model is included in the project files. The key sections are:

1. **Data Visualization**: Visualization of images from the dataset.
2. **Data Preprocessing**: Code for resizing, padding, and normalizing images.
3. **Model Training**: Training loop for VGG19 transfer learning.
4. **Model Saving**: Saving the trained model to a `.pth` file.

## Model File

The trained model is saved as `vgg19_transfer_learning_updated.pth` in the directory `D:/pythonProject/lung cancer GAN/`.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Usage

To run the code, ensure you have the required dependencies installed. Modify the paths as necessary and execute the training script to train the model and visualize the results.
