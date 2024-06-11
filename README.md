# DLStock
# Stock Market Prediction using Deep Recurrent Neural Networks


This repository contains a project for predicting stock market prices using Recurrent Neural Networks (RNN), specifically with a Bidirectional Long Short-Term Memory (LSTM) model. The project is implemented in Python and consists of two main files: `stock_csv.py` and `main.py`.


## Introduction

Stock market prediction is a crucial task for investors and financial analysts. This project aims to predict future stock prices using historical data and a Bidirectional LSTM model, which can capture patterns in the data more effectively by considering both past and future contexts.

## Requirements

To run this project, you need to have the following libraries installed:

- Python 3.9+
- pandas
- numpy
- tensorflow
- sklearn
- matplotlib
- yfinance

You can install the required libraries using pip:
```bash
pip install pandas numpy tensorflow sklearn matplotlib yfinance
```

## Project Structure
The repository contains the following files:

* **stock_csv.py** : This file includes code for loading the dataset and saving it into CSV files.
* **main.py** : This file includes code for reading the data, creating and training the Bidirectional LSTM model, and evaluating its performance.

## Dataset
The dataset used in this project consists of historical stock prices. The stock_csv.py script handles the loading and saving of this data into CSV files for further processing.
we load the historical prices using yahoo finance python library

## Model Architecture
The model used in this project is a Bidirectional LSTM, which is effective in capturing temporal dependencies in time series data. The architecture includes:

* An input layer
* Three Bidirectional LSTM layer
* Dense layers
* Dropout layers between hidden layers
* An output layer

  
## Evaluation
The performance of the model is evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The evaluation results are displayed in the output of **main.py**.
