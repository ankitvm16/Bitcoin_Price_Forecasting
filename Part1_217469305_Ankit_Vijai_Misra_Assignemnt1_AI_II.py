#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[32]:


# Function to pre-process the input data
def preprocess(data):
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


# In[33]:


# Function to split data into train and test based on a splitting date
def split(data, split_date):
  split_date = split_date
  data_train = data.loc[data.index <= split_date].copy()
  data_test = data.loc[data.index > split_date].copy()
  return(data_train, data_test)


# In[34]:


# Data Transformation: Function for min max tranform of the data
def minmaxtrans(data):
  dataset = np.reshape(data.values, (len(data.values), 1))
  sc = MinMaxScaler()
  dataset = sc.fit_transform(dataset) 
  return(dataset)


# In[35]:


# Create LSTM Model : Keras Architecture setup
def create_model():
  model = Sequential()
  model.add(LSTM(128,activation="sigmoid",input_shape=(1,1)))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  return model


# In[42]:


#load the dataset 
data = pd.read_csv('BTCUSD_hourly_data.csv')


# In[43]:


#date for splitting
split_date = '25-Jun-2018'

# Data preprocessing
data_pre= preprocess(data)
data_train, data_test = split(data_pre, split_date)
trainingset = minmaxtrans(data_train)

# Defining the X and y values for the model
X_train = trainingset[0:len(trainingset)-1]
y_train = trainingset[1:len(trainingset)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))


# In[38]:


# Compile and fit model
model = create_model()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=2)
model.summary()

# saving the trained model and model weights
model.save("forecast_model.h5")
model.save_weights("forecast_weights.h5")
print("Saved model and the weights to the disk") 


# In[44]:


#Testing the LSTM model

# load the trained model
model = load_model('forecast_model.h5')

# data transformation of the test dataset
testset = np.reshape(data_test.values, (len(data_test.values), 1))
sc = MinMaxScaler()
testset = sc.fit_transform(testset) 
X_test = np.reshape(testset, (len(testset), 1, 1))

# model prediction of test dataset
y_pred = model.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
data_test['close_prediction'] = y_pred

# model prediction using 20-point average for baseline scenario
data_test['close_prediction_ma'] = data_test['close'].rolling(20).mean()

# remove the first 20 values of the test dataset for a MSE comparison of LSTM and MA model
data_test=data_test[19:]

#Model Performance: MSE calculation
score_model= mean_squared_error(data_test['close'], data_test['close_prediction'])
print("MSE for Model using the LSTM model (Forecasting Model): %.2f" % (score_model))

score_model_ma= mean_squared_error(data_test['close'], data_test['close_prediction_ma'])
print("MSE for Model using the 20-point moving average (Baseline Model):%.2f" % (score_model_ma))

