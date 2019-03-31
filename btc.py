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