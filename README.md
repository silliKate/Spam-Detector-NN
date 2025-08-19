# Spam-Detector-NN
A deep neural network designed to classify emails and SMS messages as spam (1) or ham (0). The current implementation achieves an accuracy of approximately 95.3%.

## Overview
Spam messages, whether in emails or SMS, are a common nuisance and can sometimes pose security risks. Detecting spam automatically is crucial to improve communication efficiency and protect users from potential threats. This project implements a deep neural network for spam detection, using ReLU activation in hidden layers and a sigmoid function in the output layer for binary classification. Text data is processed using TF-IDF vectorization into numerical values, for effective learning.

## How to Run?
Pre-requisites: python
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
   
   The script preprocesses the data in spam.csv for training. You can adjust the hyperparameters as needed for experimentation purposes.
   > Note: Doing this will overwrite the existing `model_params.npz`, the original version is available in the `backup/` folder.
   
   Additionally, the script includes utility functions to monitor the model’s performance, such as tracking various learning rates vs cost during training.
   
## Learning Rate Analysis
The learning rate is a key hyperparameter that controls how much the model weights are updated during training. Choosing an appropriate learning rate is crucial:
- Too high -> model may diverge or oscillate, failing to converge.
- Too low -> training becomes very slow and may get stuck in local minima.

To find an optimal value, multiple learning rates were tested, and the corresponding cost vs. learning rate was recorded:

<p align="center">
   <img width="400" height="300" alt="Figure_train" src="https://github.com/user-attachments/assets/97a36660-1752-48e3-9267-527f9cc9f0ac" />
  <img width="400" height="300" alt="Figure_dev" src="https://github.com/user-attachments/assets/48721ce9-d803-4af5-916b-7658cacd291e" />
</p>

Learning rates 0.5 and 0.1 resulted in too low cost, signalling overfitting.
Learning rates 0.001, 0.005 and 0.0075 caused slow learning

Hence, a learning rate of 0.1 was chosen

## Receiver Operating Characteristic (ROC) curve
The Receiver Operating Characteristic (ROC) curve is a visual representation of the model’s ability to distinguish between spam and ham messages. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various classification thresholds.
- The closer the ROC curve is to the top-left corner, the better the model’s classification performance.
- The Area Under the Curve (AUC) quantifies overall performance: a value closer to 1 indicates a highly accurate model.

<p align="center">
  <img width="480" height="360" align="center" alt="ROC" src="https://github.com/user-attachments/assets/394f9ba6-73f1-40f9-b4bf-175ca08b1c6c" />
</p>

A high AUC ( ~ 0.99 ) demonstrates that the model achieves excellent discrimination between spam and non-spam messages, confirming its effectiveness.

## Acknowledgements 
The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data). The original dataset can be found on [UCI](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). 

It is composed of:
- [Grumbletext website](http://www.grumbletext.co.uk/): 425 manually extracted spam messages from a UK forum where users report SMS spam.
- [NUS SMS Corpus](http://www.comp.nus.edu.sg/~rpnlpir/downloads/corpora/smsCorpus/): 3,375 randomly selected legitimate ham messages out of the 10,000 collected for research at the Department of Computer Science at the National University of Singapore(NUS).
- [Caroline Tag’s PhD Thesis](http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf): A list of 450 SMS ham messages collected from the thesis.
- [SMS Spam Corpus v0.1 Big](http://www.esp.uem.es/jmgomez/smsspamcorpus/): 1,002 ham messages and 322 spam messages, publicly available.


