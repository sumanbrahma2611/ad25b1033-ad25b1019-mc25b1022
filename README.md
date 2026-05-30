# CNN-Based CIFAR-10 Image Classification with Custom Loss Function

## Introduction

When I started learning Deep Learning and Computer Vision, I wanted to build a project that would help me understand the complete workflow of an image classification system rather than simply training a model and observing its accuracy.

Most beginner projects focus only on achieving a good accuracy score, but I was interested in understanding how images are processed, how convolutional neural networks learn patterns, how loss functions influence learning, and how different evaluation metrics help measure model performance.

To explore these concepts, I built a CNN-based image classification model using the CIFAR-10 dataset. In addition to the standard training process, I experimented with a custom loss function designed to reduce overconfident predictions and improve model generalization. I also extended the project by including Precision, Recall, F1-Score, Classification Reports, prediction visualization, and training performance graphs.

---

# Project Motivation

The primary motivation behind this project was to gain practical experience in Deep Learning by implementing an image classification system from scratch.

Through this project, I wanted to understand:

* How Convolutional Neural Networks (CNNs) extract features from images.
* How custom loss functions affect model training.
* Techniques used to reduce overfitting.
* The importance of evaluating a model using multiple metrics.
* The complete machine learning workflow from dataset preparation to model evaluation.

This project helped me strengthen my understanding of Computer Vision fundamentals while working on a real-world image classification problem.

---

# Problem Statement

Image classification is one of the most important applications of Computer Vision.

Humans can easily recognize whether an image contains a cat, dog, airplane, or automobile. However, for machines, images are simply collections of numerical pixel values. The challenge is to train a model that can automatically learn meaningful visual features from images and accurately classify them into predefined categories.

The objective of this project is to build a deep learning model capable of classifying images into ten different categories using the CIFAR-10 dataset.

---

# Why CIFAR-10?

The CIFAR-10 dataset was chosen because:

* It is one of the most widely used benchmark datasets in Computer Vision.
* It contains real-world colored images instead of simple handwritten digits.
* It is beginner-friendly while still providing meaningful classification challenges.
* It allows experimentation with CNN architectures without requiring extremely high computational resources.
* It is commonly used for learning image classification techniques and comparing model performance.

---

# Dataset Description

CIFAR-10 is a collection of small colored images grouped into ten different classes.

### Dataset Information

| Property          | Value       |
| ----------------- | ----------- |
| Training Images   | 50,000      |
| Testing Images    | 10,000      |
| Number of Classes | 10          |
| Image Size        | 32 × 32 × 3 |
| Dataset Type      | RGB Images  |

### Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

---

# Technologies Used

The following technologies and libraries were used during development:

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-Learn

---

# Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of multiple feature extraction and classification layers.

### Convolution Block 1

* Conv2D (32 Filters)
* Batch Normalization
* Max Pooling
* Dropout

### Convolution Block 2

* Conv2D (64 Filters)
* Batch Normalization
* Max Pooling
* Dropout

### Convolution Block 3

* Conv2D (128 Filters)
* Batch Normalization
* Max Pooling
* Dropout

### Classification Layers

* Flatten Layer
* Dense Layer (256 Neurons)
* Dropout
* Output Layer (10 Neurons with Softmax Activation)

---

# Custom Loss Function

Instead of using only the standard categorical cross-entropy loss, a custom loss function was implemented.

### Formula

Loss = Cross Entropy + λ × Confidence Penalty

Where:

* Cross Entropy measures classification error.
* Confidence Penalty discourages extremely confident predictions.
* λ = 0.1 acts as a balancing factor.

### Purpose

The custom loss function was introduced to:

* Reduce overconfident predictions.
* Encourage better probability distributions.
* Improve generalization performance.
* Make the model less prone to overfitting.

---

# Features

This project includes:

* Custom Loss Function
* CNN-Based Architecture
* Batch Normalization
* Dropout Regularization
* Accuracy Evaluation
* Precision Evaluation
* Recall Evaluation
* F1-Score Evaluation
* Classification Report
* Sample Prediction Visualization
* Training and Validation Accuracy Graphs

---

# Project Workflow

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Image Normalization
        ↓
CNN Model Creation
        ↓
Custom Loss Function
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction Generation
        ↓
Performance Analysis
        ↓
Visualization
```

---

# Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git

cd your-repository-name
```

## Step 2: Create a Virtual Environment

### Linux / WSL

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

# Running the Project

## Method 1: Using VS Code

### Step 1

Open the project folder in VS Code.

### Step 2

Open the integrated terminal.

### Step 3

Install required dependencies.

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Step 4

Run the program.

```bash
python cifar10_cnn_classifier.py
```

### Output

The program will display:

* Model Summary
* Training Progress
* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Sample Predictions
* Accuracy Graph

---

## Method 2: Using WSL

### Step 1

Open Ubuntu Terminal.

### Step 2

Navigate to the project directory.

```bash
cd path/to/project
```

### Step 3

Install dependencies.

```bash
pip3 install tensorflow numpy matplotlib scikit-learn
```

### Step 4

Run the project.

```bash
python3 cifar10_cnn_classifier.py
```

---

## Method 3: Using Google Colab

### Step 1

Open Google Colab.

### Step 2

Upload the Python file.

### Step 3

Install required libraries if necessary.

```python
!pip install tensorflow numpy matplotlib scikit-learn
```

### Step 4

Run the script.

```python
!python cifar10_cnn_classifier.py
```

### Output

Google Colab will automatically:

* Download CIFAR-10
* Train the model
* Evaluate the model
* Display metrics
* Show prediction examples
* Plot accuracy graphs

---

# Performance Metrics

To obtain a complete understanding of model performance, multiple evaluation metrics are used.

### Accuracy

Measures overall prediction correctness.

### Precision

Measures how many predicted labels are actually correct.

### Recall

Measures how many actual labels were successfully identified.

### F1 Score

Balances Precision and Recall into a single metric.

### Classification Report

Provides class-wise performance analysis.

---

# Sample Output

```text
================================
MODEL PERFORMANCE
================================

Test Accuracy : XX.XX%

Precision : 0.XXXX

Recall : 0.XXXX

F1 Score : 0.XXXX

CLASSIFICATION REPORT

<Classification Report Output>

SAMPLE PREDICTIONS

Image 1

Actual Label    : Dog
Predicted Label : Dog

Image 2

Actual Label    : Cat
Predicted Label : Cat
```

---

# Future Improvements

Possible enhancements include:

* Data Augmentation
* Transfer Learning
* MobileNet Architecture
* ResNet Architecture
* EfficientNet Architecture
* Learning Rate Scheduling
* Hyperparameter Optimization
* Confusion Matrix Visualization
* Streamlit Deployment
* Flask API Deployment

---

# Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning Fundamentals
* Convolutional Neural Networks
* Image Classification
* TensorFlow and Keras
* Data Preprocessing
* Custom Loss Function Design
* Model Evaluation Techniques
* Performance Metrics Analysis
* Computer Vision Workflows

---

# Conclusion

This project demonstrates how Convolutional Neural Networks can learn meaningful visual patterns from images and classify them into multiple categories. Beyond achieving classification performance, the project focuses on understanding the complete deep learning workflow, including preprocessing, model design, custom loss implementation, training, evaluation, and visualization.

The project served as a valuable learning experience and provides a strong foundation for exploring more advanced Computer Vision and Deep Learning architectures in the future.

---

# Author

**Suman Brahma, Vijay Verma, Mohd Faizan Khan**

If you found this project useful, feel free to star the repository and share your feedback.
