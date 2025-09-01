# CNN-Based-Sleep-Stage-Classification
This notebook implements a Convolutional Neural Network (CNN) model for sleep stage classification using EEG datasets. It preprocesses raw EEG signals, extracts relevant features, and trains CNN models on multiple datasets (Sleep-EDF, Labaid, Sleep Bank). The workflow integrates signal processing techniques and deep learning for biomedical applications.\
.

🔑 Step-by-Step Breakdown

Environment Setup

Connects Google Drive for dataset access (drive.mount).

Installs required libraries: mne (EEG data analysis).

Import Libraries

Core: numpy, pandas, os.

Signal Processing: scipy.signal (resampling, filtering).

EEG Processing: mne.

ML/DL: sklearn (train-test split), tensorflow.keras (CNN model).

Data Loading

Loads EEG sleep datasets:

Sleep Bank data

Labaid hospital data

Handles raw signals (.edf format) using mne.

Preprocessing

Signal filtering with FIR filter and filtfilt.

Resampling signals for consistency.

Splitting into epochs for training.

Feature scaling and normalization.

Dataset Variants

Separate training pipelines for:

Sleep Bank dataset

Labaid dataset

Enables model performance comparison across datasets.

CNN Model Architecture

Input: EEG signal epochs.

Convolutional layers for feature extraction.

Pooling layers for dimensionality reduction.

Fully connected layers leading to classification output.

Output: Predicted sleep stage categories (e.g., REM, NREM, Wake).

Training

Splits data into training & validation sets.

Uses categorical crossentropy loss.

Optimizer: likely Adam (standard for CNNs).

Evaluation metrics: Accuracy, possibly Precision/Recall.

Evaluation

Model tested on both Sleep Bank and Labaid datasets.

Accuracy and confusion matrix (likely included in results).

Comparisons across datasets to check generalization.
