# Benign-vs-Malignant-Classification
Breast cancer screening requires accurate distinction between benign and malignant lesions. This project uses cropped mammogram ROIs from the datasets to train deep learning models. Pretrained CNNs are fine-tuned for binary classification with augmentation. The goal is to provide a solid baseline with AUROC ≥ 0.85 for reliable lesion diagnosis.

# Benign vs Malignant Classification from Cropped Mammogram Lesions

## Problem Statement

Breast cancer diagnosis depends on accurate differentiation between benign and malignant lesions. This project focuses on training a deep learning model to classify cropped regions of interest (ROIs) from mammograms as benign or malignant using the CBIS-DDSM dataset.

## Description

Breast cancer screening requires accurate distinction between benign and malignant lesions. This project uses cropped mammogram ROIs from the CBIS-DDSM dataset to train deep learning models. Pretrained CNNs (EfficientNet, DenseNet) are fine-tuned for binary classification with augmentation. The goal is to provide a solid baseline with AUROC ≥ 0.85 for reliable lesion diagnosis.

## Data

The project uses the `mass_case_description_train_set.csv` file from the CBIS-DDSM dataset. This dataset contains information about mass lesions in mammograms, including patient details, breast density, abnormality characteristics (shape, margins, assessment), and image file paths.

## Project Steps

The notebook covers the following steps:

1.  **Import Libraries**: Imports necessary libraries for data manipulation, machine learning, and deep learning.
2.  **Load Data**: Loads the `mass_case_description_train_set.csv` file into a pandas DataFrame.
3.  **Data Description**: Provides an initial overview of the data, including the first few rows, data types, summary statistics, and unique pathology labels.
4.  **Data Cleaning & Filtering**: Normalizes the 'pathology' column and filters the DataFrame to include only 'benign' and 'malignant' cases.
5.  **Feature Selection & Encoding**: Selects relevant features and encodes categorical features using Label Encoding. The target variable 'pathology' is also encoded.
6.  **Split Data & Scale**: Splits the data into training and testing sets and scales the 'assessment' feature using StandardScaler.
7.  **Train Model & Evaluate (Random Forest)**: Implements and evaluates a RandomForestClassifier model.
8.  **Train Model & Evaluate (Basic Neural Network)**: Implements and evaluates a basic Sequential Neural Network model using TensorFlow.
9.  **Train Model & Evaluate (Improved Neural Network)**: Implements and evaluates an improved Sequential Neural Network model with Batch Normalization and Dropout layers, and Early Stopping.
10. **Hyperparameter Tuning (Keras Tuner - Basic)**: Uses Keras Tuner's RandomSearch to find optimal hyperparameters for a neural network model with embeddings for categorical features.
11. **Hyperparameter Tuning (Keras Tuner - Advanced)**: Explores a wider range of hyperparameters and optimization settings for the neural network model using Keras Tuner.
12. **Stacked Ensemble Model**: Builds a stacking ensemble model combining XGBoost, RandomForest, and the tuned Keras model. This section also includes Stratified K-Fold Cross-Validation and confusion matrix visualization for each fold.

## Setup and Usage

### Prerequisites

*   Python 3.7+
*   Jupyter Notebook or Google Colab
*   Required libraries: pandas, scikit-learn, tensorflow, xgboost, keras-tuner, matplotlib, seaborn

### Installation

1.  Clone the repository (if applicable).
2.  Install the required libraries:
