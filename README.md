# Retinal Disease Classification using Deep Learning (OCTMNIST)

This project implements and analyzes deep learning models for retinal disease classification using the OCTMNIST dataset (MedMNIST collection).

The objective is to design, train, optimize, interpret, and deploy a convolutional neural network capable of classifying OCT images into 4 disease categories.

---

## Dataset

The OCTMNIST dataset consists of 2D grayscale optical coherence tomography (OCT) images used for retinal disease classification. Each image has a size of 28 × 28 pixels with 1 channel. The dataset contains 97,477 training samples, 10,832 validation samples, and 1,000 test samples, which are split among 4 classes: choroidal neovascularization, diabetic macular edema, drusen, and normal retina.
Images were normalized to [0,1]. Class distribution was analyzed and handled where necessary.

---

## Base Model Architecture

* This model is a Convolutional Neural Network (CNN) designed for classifying OCT images into 4 classes.
* The input has 1 channel (grayscale image).
* Based on the architecture (128 × 3 × 3 before flattening), the input image size is 28 × 28 pixels.
* There are 4 output neurons, one for each class.
* For the hidden layers, ReLU (Rectified Linear Unit) activation is used.
* There is no activation function applied in the final layer. This is because CrossEntropyLoss is used during training, which internally applies Softmax by itself.
* The model has:
	*	6 convolutional hidden layers
	*	1 fully connected hidden layer (256 neurons)
* The size of he hidden layers is as below
	*	Conv layers: 32 filters → 64 filters → 128 filters
	*	Fully connected hidden layer: 256 neurons
* A Dropout layer with 0.3 probability is used in the fully connected part to reduce overfitting.


## Improvement Techniques Applied:
- Early Stopping
- Learning Rate Scheduler
- Batch Normalization
- Data Augmentation
- Dropout Regularization

Final Test Accuracy: **77.40%**
Improved Model Accuracy: **82.60%**

---

## Results

The improved model achieved over 80% test accuracy.

---

## 🔍 Model Interpretability

To understand model behavior:
- Visualized convolutional feature maps
- Captured activation outputs using PyTorch hooks
- Visualized first-layer convolutional kernels

In the early layers, the network mainly highlights the bright retinal layer boundaries and strong horizontal edges, while darker background areas are mostly suppressed.

In the deeper layers, it focuses more on specific patterns like layer thickness changes and abnormal bright regions. It ignores the smooth or uniform areas that are less important.

---

## Deployment

The model was deployed locally using Streamlit.

The app allows users to upload OCT images and receive disease predictions in real time.

---

## Tech Stack

- Python
- PyTorch
- Scikit-learn
- Matplotlib / Seaborn
- Torchinfo
- Streamlit (Deployment)

