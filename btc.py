# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:20:11 2019

@author: ALEXK
"""

import requests,json,numpy as np,pandas as pd
#https://api.coinranking.com/v1/public/coin/:coin_id/history/:timeframe
#https://docs.coinranking.com/
def hist_price_dl(coin_id=1335,timeframe = "5y",currency = "USD"):
    '''It accepts coin_id, timeframe, and currency parameters to clean the historic coin data taken from COINRANKING.COM
    It returns a Pandas Series with daily mean values of the selected coin in which the date is set as the index'''
    r = requests.get("https://api.coinranking.com/v1/public/coin/"+str(coin_id)+"/history/"+timeframe+"?base="+currency)
    coin = json.loads(r.text)['data']['history'] #Reading in json and cleaning the irrelevant parts
    df = pd.DataFrame(coin)
    df['price'] = pd.to_numeric(df['price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms').dt.date
    return df.groupby('timestamp').mean()['price']

def price_matrix_creator(data, seq_len=30):
    '''
    It converts the series into a nested list where every item of the list contains historic prices of 30 days
    '''
    price_matrix = []
    for index in range(len(data)-seq_len+1):
        price_matrix.append(data[index:index+seq_len])
    return price_matrix

def zero_division(n, d):        #Used because I was getting divide by 0 errors in normalize method
    return n / d if d else 0

def normalize_windows(window_data):
    '''
    It normalizes each value to reflect the percentage changes from starting point
    '''
    normalised_data = []
    for window in window_data:
        normalised_window = [(zero_division(float(p), float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def train_test_split_(price_matrix, train_size=0.9, shuffle=False, return_row=True):
    '''
    It makes a custom train test split where the last part is kept as the training set.
    '''
    price_matrix = np.array(price_matrix)
    #print(price_matrix.shape)
    row = int(round(train_size * len(price_matrix)))
    train = price_matrix[:row, :]
    if shuffle==True:
        np.random.shuffle(train)
    X_train, y_train = train[:row,:-1], train[:row,-1]
    X_test, y_test = price_matrix[row:,:-1], price_matrix[row:,-1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if return_row:
        return row, X_train, y_train, X_test, y_test
    else:
        X_train, y_train, X_test, y_test


ser = hist_price_dl() # Not passing any argument since they are set by default
price_matrix = price_matrix_creator(ser) # Creating a matrix using the dataframe
price_matrix = normalize_windows(price_matrix) # Normalizing its values to fit to RNN
row, X_train, y_train, X_test, y_test = train_test_split_(price_matrix) # Applying train-test splitting, also returning the splitting-point

#Time to build RNN

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import time
# LSTM Model parameters, I chose
batch_size = 2            # Batch size (you may try different values)
epochs = 15               # Epoch (you may try different values)
seq_len = 30              # 30 sequence data (Representing the last 30 days)
loss='mean_squared_error' # Since the metric is MSE/RMSE
optimizer = 'rmsprop'     # Recommended optimizer for RNN
activation = 'linear'     # Linear activation
input_shape=(None,1)      # Input dimension
output_dim = 30           # Output dimension

model = Sequential()
model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
model.add(Dense(units=32,activation=activation))
model.add(LSTM(units=output_dim, return_sequences=False))
model.add(Dense(units=1,activation=activation))
model.compile(optimizer=optimizer,loss=loss)

start_time = time.time()
model.fit(x=X_train,
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05)
end_time = time.time()
processing_time = end_time - start_time


model.save('coin_predictor.h5')

import requests,json,numpy as np,pandas as pd
#We need ser, preds, row
ser = hist_price_dl(timeframe='30d')[1:31]
price_matrix = price_matrix_creator(ser)
X_test = normalize_windows(price_matrix)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

from keras.models import load_model
model = load_model('coin_predictor.h5')
preds = model.predict(X_test, batch_size=2)

def deserializer(preds, data, train_size=0.9, train_phase=False):
    '''
    Arguments:
    preds : Predictions to be converted back to their original values
    data : It takes the data into account because the normalization was made based on the full historic data
    train_size : Only applicable when used in train_phase
    train_phase : When a train-test split is made, this should be set to True so that a cut point (row) is calculated based on the train_size argument, otherwise cut point is set to 0
    
    Returns:
    A list of deserialized prediction values, original true values, and date values for plotting
    '''
    price_matrix = np.array(price_matrix_creator(ser))
    if train_phase:
        row = int(round(train_size * len(price_matrix)))
        print(row)
    else:
        row=0
    date = ser.index[row+29:]
    date = np.reshape(date, (date.shape[0]))
    X_test = price_matrix[row:,:-1]
    y_test = price_matrix[row:,-1]
    preds_original = []
    preds = np.reshape(preds, (preds.shape[0]))
    for index in range(0, len(preds)):
        pred = (preds[index]+1)* X_test[index][0]
        preds_original.append(pred)
    preds_original = np.array(preds_original)
    if train_phase:
        return [date, y_test, preds_original]
    else:
        import datetime
        return [date+datetime.timedelta(days=1),y_test]


final_pred = deserializer(preds, ser, train_size=0.9, train_phase=False)
final_pred[1][0]

from keras.models import load_model
model = load_model('coin_predictor.h5')
preds = model.predict(X_test, batch_size=2)
plotlist = deserializer(preds, ser, train_phase=True)


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

prices = pd.DataFrame({'Predictions':plotlist[1], 'Real Prices':plotlist[2]},index=plotlist[0])
iplot(prices.iplot(asFigure=True,
                   kind='scatter',
                   xTitle='Date',
                   yTitle='BTC Price',
                   title='BTC Price Predictions'))