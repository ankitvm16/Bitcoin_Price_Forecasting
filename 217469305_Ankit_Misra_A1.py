#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Importing the libraries
import numpy as np 
import pandas as pd
import datetime
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import load_model


# In[33]:


# Function which takes a CSV file as input from the user and preprocesses the data
# to return a numpy array of input to the model (X) and another numpy array of target value (y)

def preprocess(CSVfilename):
    
    filename= CSVfilename+".csv"
    data = pd.read_csv(filename)
    # data pre-processing of the validation dataset
    data_pre= pretreat(data)
    
    # data transformation of the validation dataset
    X = np.reshape(data_pre.values, (len(data_pre.values), 1))
    y_df= data_pre[1:len(data_pre)]
    last=y_df.iloc[-1]
    y_df.append(last)
    y= y_df.to_numpy()
    return(X,y)


# In[42]:


# Function with the Model architecture which takes 'filename of the model weights', and X
# (returned by preprocess() func as the argument and returns a numpy array of predictions.

# the model also calculates the 20-point moving average prediction and provides the MSE
# of the LSTM forecasting model and 20-point MA model for comparison

def model(file_weights, X):
    data_valid=pd.DataFrame(X, columns=['close'])
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X = np.reshape(X, (len(X), 1, 1))
    
    # Create LSTM Model : Keras Architecture setup
    model = Sequential()
    model.add(LSTM(128,activation="sigmoid",input_shape=(1,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Load trained model weights
    filename= file_weights+".h5"
    model.load_weights(filename)
    
    #LSTM model forecasting of the future values
    pred=model.predict(X)
    y_pred = sc.inverse_transform(pred)
    data_valid['close_prediction'] = pd.DataFrame(y_pred)
    
    # 20-point moving average model forecasting of the future values
    data_valid['close_prediction_ma'] = data_valid['close'].rolling(20).mean()
    
    # remove the first 20 values of the test dataset for a MSE comparison of LSTM and MA model
    data_valid=data_valid[19:]
    
    # plotting the forecasted and the actual values for comparitive analysis
    plt.figure(figsize=[20,8])
    plt.title('BTC Actual and Forecasted Closing Price (USD) for the validation dataset')
    plot1 = plt.plot(data_valid['close'], label='Actual Price')
    plot2 = plt.plot(data_valid['close_prediction_ma'], label='MA Forecasted Price')
    plot3 = plt.plot(data_valid['close_prediction'], label='LSTM Forecasted Price')
    #plt.legend([plot1,plot2, plot3],["Actual Price", "MA Forecasted Price", "LSTM Forecasted Price" ])
    plt.ylabel("Price")
    plt.legend(loc=2, fontsize="small")
    plt.xticks([])
    #plt.xlabel("Dates")
    plt.show()
    
    #Model Performance: MSE calculation
    score_model= mean_squared_error(data_valid['close'], data_valid['close_prediction'])
    print("MSE for Model using the LSTM model (Forecasting Model): %.2f" % (score_model))
    score_model_ma= mean_squared_error(data_valid['close'], data_valid['close_prediction_ma'])
    print("MSE for Model using the 20-point moving average (Baseline Model):%.2f" % (score_model_ma))
    
    return(y_pred)


# In[35]:


# Function to pre-process the input data
def pretreat(data):
  # Missing Value Treatment: the OHLC (open high low close) data is a continuous timeseries hence filled with fill forwards values.
  data['open'].fillna(method='ffill', inplace=True)
  data['high'].fillna(method='ffill', inplace=True)
  data['low'].fillna(method='ffill', inplace=True)
  data['close'].fillna(method='ffill', inplace=True)
    # volume is a single event and hence NA's are replaced with zeroes
  data['volume'].fillna(value=0, inplace=True)

  # changing to datetime and index assignment to date variable
  data['date'] = pd.to_datetime(data['date'])
  data = data.groupby([pd.Grouper(key='date', freq='H')]).first().reset_index()
  data = data.set_index('date')
  data = data[['close']]
  return(data)


# In[36]:


# Data Transformation: Function for min max tranform of the data
def minmaxtrans(data):
  dataset = np.reshape(data.values, (len(data.values), 1))
  sc = MinMaxScaler()
  dataset = sc.fit_transform(dataset) 
  return(dataset)


# In[43]:


# Main script to call the requisite functions
# Takes user input for the following:
# a) Validation dataset CSV file name
# b) model weights filename
print("This is a Bitcoin Forecasting Model")
print("The model predicts the hourly closing price")
print("Please specify the filename containing the OHLC-Vol data with hourly timestamp")
CSVfilename = input("Enter the Validation dataset CSV file name : ")
#file_weights = input("Input the model weights filename: ")
file_weights = "forecast_weights"
X, y = preprocess(CSVfilename)
y_pred = model(file_weights, X)
print ("The forecasted closing price of BTC for the next hour is %.2f" % (y_pred[-1]))


# In[ ]:




