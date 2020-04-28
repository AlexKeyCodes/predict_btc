# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:07:51 2019

@author: ALEXK
"""

import json
import requests

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

sns.set_palette('Set2')


endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
hist = pd.DataFrame(json.loads(res.content)['Data'])
y = hist.iloc[:, 4].values
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
y = pd.to_datetime(y, unit='s')
hist.head()
X = hist.iloc[:, :].values
y = hist.iloc[:, 1].values


def train_test_split(df, test_size):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def line_plot(line1, line2, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    
#train test split of 80/20
train, test = train_test_split(hist, test_size=0.2)
line_plot(train.close, test.close, 'training', 'test', 'BTC')


def normalise_zero_base(df):
    """ First index is 0, the others are the difference in values
        from the first value
    """
    return df / df.iloc[0] - 1

def extract_window_data(df, window=7, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of
        length `window`. Using 7 to simulate 1 weeks worth of data
    """
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, window=7, zero_base=True, test_size=0.2):
    """ Prepare data for model """
    # train test split
    train_data, test_data = train_test_split(df, test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)
    
    # extract targets
    y_train = train_data.close[window:].values
    y_test = test_data.close[window:].values
    if zero_base:
        y_train = y_train / train_data.close[:-window].values - 1
        y_test = y_test / test_data.close[:-window].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

train, test, X_train, X_test, y_train, y_test = prepare_data(hist)

""" Using 20 neurons, tanh activation, .25 dropout, mean squared error, 
    and adam optimizer. 
    Single LSTM layer and dense layer with tanh activation function """
def build_lstm_model(input_data, output_size, neurons=20,
                     activ_func='linear', dropout=0.25,
                     loss='mean_squared_error', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(
              input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

model = build_lstm_model(X_train, output_size=1)
history = model.fit(X_train, y_train, epochs=10, batch_size=4)

target_col = 'close'
targets = test[target_col][7:]
preds = model.predict(X_test).squeeze()
# convert change predictions back to actual price
preds = test.close.values[:-7] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
n = 30 #1month
line_plot(targets[-n:], preds[-n:], 'actual', 'prediction')
#shift by one day
line_plot(targets[-n:][:-1], preds[-n:].shift(-1), 'actual', 'prediction')
n = 365 #1 year
line_plot(targets[-n:], preds[-n:], 'actual', 'prediction')
#shift by one day
line_plot(targets[-n:][:-1], preds[-n:].shift(-1), 'actual', 'prediction')


actual_returns = targets.pct_change()[1:]
predicted_returns = preds.pct_change()[1:]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
# actual correlation
corr = np.corrcoef(actual_returns, predicted_returns)[0][1]
ax1.scatter(actual_returns, predicted_returns, color='k')
ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)
# shifted correlation
shifted_actual = actual_returns[:-1]
shifted_predicted = predicted_returns.shift(-1).dropna()
corr = np.corrcoef(shifted_actual, shifted_predicted)[0][1]
ax2.scatter(shifted_actual, shifted_predicted, color='k')
ax2.set_title('r = {:.2f}'.format(corr));



