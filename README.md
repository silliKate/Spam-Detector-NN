# Spam-Detector-NN
A deep neural network designed to classify emails and SMS messages as spam or ham. The current implementation achieves an accuracy of approximately 95.3%.

## Overview
Spam messages, whether in emails or SMS, are a common nuisance and can sometimes pose security risks. Detecting spam automatically is crucial to improve communication efficiency and protect users from potential threats. This project implements a deep neural network for spam detection, using ReLU activation in hidden layers and a sigmoid function in the output layer for binary classification. Text data is processed using TF-IDF vectorization into numerical values, for effective learning.

## How to Run?
1. Clone the repository using

   ```
   git clone https://github.com/silliKate/Spam-Detector-NN
   ```

   This will download all the project files, including scripts, datasets, and model parameters.
   
2. Install the requirements

   ```
   pip install -r requirements.txt
   ```
   
   These packages include NumPy, pandas, Matplotlib, and scikit-learn, which are necessary for data processing, model training, and visualization.
  
3. Test the model on custom messages
   
   You can test the model without retraining by running:
   
   ```
   python test.py
   ```
   
   You will be prompted to enter a message. The model will predict whether the message is spam or not. It will also display a confidence score indicating how certain it is about the prediction.

5. Train or retrain the model
   
   ```
   python train.py
   ```
   
   The script preprocesses the data in spam.csv for training. You can adjust the hyperparameters as needed for experimentation or testing.
   Note: Running this script will overwrite the existing model_params.npz; the original version is available in the backup/ folder.
   Additionally, the script includes utility functions to monitor the model’s performance, such as tracking various learning rates vs cost during training.
   
