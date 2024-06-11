import pandas as pd
import keras.losses
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import tensorflow as tf

tf.config.run_functions_eagerly(True)

def future_predict(last_feature, model, days):
    """
    :param last_feature: is numpy array 2D
    :param model: tensorflow LSTM model
    :param days: how many days you want to predict in future
    :return: a list of future days prices

    """
    listed_array = list(last_feature)
    listed_array = [listed_array] # make it 3D
    future_temp = np.array(listed_array)
    future_predictions = np.array([])

    for i in range(days):
        temp_predict = model.predict(future_temp)
        temp_predict = temp_predict[0]
        future_predictions = np.append(future_predictions, temp_predict)
        temp = list(future_temp[0])
        temp = temp[1:]
        temp.append(list(temp_predict))
        temp = [temp]
        future_temp = np.array(temp)

    return future_predictions


def x_y_train(the_list, days):
    n = days

    x_train = [the_list[i:i+n] for i in range(len(the_list)-n)]
    y_train = [the_list[i][0] for i in range(n, len(the_list))]

    return np.array(x_train), np.array(y_train)


plt.style.use('fivethirtyeight')


company = input('ENTER THE NAME OF COMPANY : ')
data = pd.read_csv(f'Data/{company}.csv')



close = data.filter(['Close'])
dataset = close.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

days = int(input('Based on how many days you want to train the model ? : '))
x_train , y_train = x_y_train(scaled_data, days)


epoch_number = int(input('ENTER THE EPOCHS : '))

x_train, y_train = x_train[2000:], y_train[2000:]

# Creating Recurrent Model
model = Sequential([

    Bidirectional(LSTM(256, return_sequences= True, input_shape= (x_train.shape[1], 1), activation='tanh')),
    keras.layers.Dropout(0.2),
    Bidirectional(LSTM(512, return_sequences= True, activation='tanh')),
    keras.layers.Dropout(0.2),
    Bidirectional(LSTM(128, return_sequences= False, activation='tanh')),
    keras.layers.Dropout(0.2),
    Dense(128, activation='elu'),
    keras.layers.Dropout(0.2),
    Dense(64, activation='elu'),
    Dense(128, activation='tanh'),
    keras.layers.Dropout(0.2),
    Dense(32, activation='elu'),
    Dense(1, activation='linear')


])


model.compile(optimizer='adam', loss = 'mse' , run_eagerly=True)


model.fit(x_train, y_train, batch_size=1, epochs=epoch_number)


order = input('past prediction or future ? : ')

while order != 'exit' :

    if order == 'future':
        days = int(input('ENTER THE DAYS IN FUTURE : '))

        predictions = future_predict(x_train[-1], model, days)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        dataset = [item[0] for item in dataset]
        predictions = [item[0] for item in predictions]
        whole = dataset + predictions

        """
        ploting the data predicted in future
        """

        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('close AAPL USD$')
        plt.plot(whole)
        plt.legend(['Actual', 'Predictions'])

        plt.show()

        order = input('past prediction or future ? : ')


    elif order == 'past':
        n = int(input('enter the days from past till now : '))
        actual = y_train[-n::]
        predicted_n = model.predict(x_train[-n::])

        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('close AAPL USD$')
        plt.plot(actual)
        plt.plot(predicted_n)

        plt.legend(['Actual', 'Predictions'])

        plt.show()

        order = input('past prediction or future ? : ')

