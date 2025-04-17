# CIFAR-10 Image Classification Using Deep Neural Networks (DNN / MLP)

This project demonstrates image classification on the CIFAR-10 dataset using a Deep Neural Network (DNN), also known as a Multi-Layer Perceptron (MLP), without using any Convolutional Neural Networks (CNNs). The key objective of this case study is to classify RGB images into one of ten predefined categories, leveraging fully connected layers and hyperparameter tuning.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
  - [Model Summary](#model-summary)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Training](#training)
- [Results](#results)
- [Challenges](#challenges)
- [Technologies Used](#technologies-used)
- [Conclusion](#conclusion)

---

## Overview

This notebook walks through the complete pipeline for training a deep learning model using DNNs for image classification, starting from data preprocessing to model training and evaluation. The focus is on tuning the model using Keras Tuner to achieve optimal architecture and training configuration.

---

## Dataset

The CIFAR-10 dataset consists of 60,000 color images of size 32x32 pixels in 10 classes:

- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

It includes 50,000 training images and 10,000 test images.

---

## Preprocessing

### Normalization
- Pixel values were scaled to the range [0, 1] by dividing by 255.0.

### Label Encoding
- Labels were one-hot encoded for multi-class classification.

### Label Mapping
- Numerical labels were mapped to class names for visualization.

---

## Model Architecture

- **Input Layer**: Flattened 32x32x3 images to 1D vectors (3072 features).  
- **Hidden Layers**: Dynamically constructed based on hyperparameters.  
- **Activation Functions**: ReLU, Sigmoid, Tanh, LeakyReLU.  
- **Regularization**: L2 regularization applied to all dense layers.  
- **Dropout**: Applied to prevent overfitting.  
- **Output Layer**: Dense layer with 10 neurons and Softmax activation.  

> **Note:** No convolutional layers were used to emphasize MLP performance on image data.

---

### Model Summary
### Model Summary

| Layer (type)        | Output Shape   | Param # |
|---------------------|----------------|---------|
| Flatten             | (None, 3072)   | 0       |
| Dense               | (None, 288)    | 885,024 |
| BatchNormalization  | (None, 288)    | 1,152   |
| Dropout             | (None, 288)    | 0       |
| Dense               | (None, 10)     | 2,890   |
| **Total Parameters**     |                | **889,066** |
| **Trainable Params**     |                | **888,490** |
| **Non-trainable Params** |                | **576**     |

Total Parameters: 889,066
Trainable Params: 888,490
Non-trainable Params: 576

### Conclusion

This project demonstrates the application of Deep Neural Networks (DNNs), also known as Multi-Layer Perceptrons (MLPs), for image classification on the CIFAR-10 dataset. While MLPs can be used for image classification, they struggle to capture spatial features as effectively as Convolutional Neural Networks (CNNs), which are typically the go-to architecture for image-related tasks.

Despite extensive hyperparameter tuning using **Keras Tuner**, the best validation accuracy achieved was **50.7%**, which highlights the inherent limitations of MLPs in handling image data without the spatial hierarchies that CNNs provide. 

#### Key Insights:
- **Best Validation Accuracy**: 50.7% (achieved using 1 Hidden Layer with 288 neurons, Tanh activation, SGD optimizer with a learning rate of 0.00023, and 0.3 dropout).
- **Model Performance**: The performance of MLPs on image data remains relatively low compared to CNNs, which excel in tasks involving spatial data.
- **Hyperparameter Tuning**: Hyperparameters such as the number of neurons, layers, activation functions, and optimizer choices were extensively tuned using Keras Tuner. However, even with careful tuning, the performance did not exceed 50.7%.
- **Overfitting**: Techniques like **dropout** (0.3) and **early stopping** were applied to prevent overfitting, though challenges remain, as the MLPs are still prone to underfitting or poor generalization for image data.

#### Evaluation Metrics:
- **Best Model Configuration**:
    - **Number of Hidden Layers**: 1
    - **Number of Neurons in Hidden Layer**: 288
    - **Activation Function**: Tanh
    - **Optimizer**: SGD with learning rate of 0.00023
    - **Dropout**: 0.3
    - **Kernel Initializer**: Glorot Uniform
- **Training Accuracy**: ~80.5% (achieved during training before validation).
- **Validation Accuracy**: ~50.7% (best result on the validation set).

#### Challenges:
- **MLPs** are not naturally designed for image classification tasks, as they lack mechanisms to learn spatial relationships in images.
- Even after hyperparameter optimization, achieving high accuracy on image classification tasks using DNN alone remains a challenge.
- **Overfitting** was mitigated using dropout and early stopping, but these techniques couldn't entirely overcome the limitations of MLPs for this type of data.

#### Conclusion:
While **MLPs** can be used for image classification, they are not the most effective architecture when compared to **Convolutional Neural Networks (CNNs)**. This project reinforces the understanding that CNNs are the preferred choice for image-related tasks due to their ability to efficiently capture spatial hierarchies and patterns in images. 

The project provides insights into the **limitations of fully connected networks** for image classification and showcases the effectiveness of **hyperparameter tuning** in improving model performance. However, future work can explore more advanced architectures, such as **CNNs**, to achieve better performance on image classification tasks.

This experiment serves as a valuable learning exercise in understanding how DNNs behave with image data, and emphasizes the need for **CNN-based architectures** for achieving state-of-the-art results in image classification.



