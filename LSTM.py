# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:14:04 2017

@author: Admin
"""
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from random import randint


# load dataset
loaddata = read_csv('sensor-data.csv')
set1 = loaddata[['Nr-People','Time','Noise','Brightness','Relative-humidity','Motion']]
set1['Time'] = set1['Time'].str.replace(':','.') #used to convert : to . to make time a float data
set1['Time'] = set1['Time'].astype(float) 
set1 = set1.astype('float32')
print(set1.head())
values = set1.values
x=values[:,1:]
y=values[:,0]


#split data into train and test set exactly followed as the tutorial from internet test size 30 % 
x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
train_X = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
test_X = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
 
# design network
model = Sequential()
model.add(LSTM(80,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1, activation='relu'))
print(model.summary())
model.compile(loss='mae', optimizer='adam')

# fitting of the network
history = model.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Test')
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 5
fig_size[1] = 3
plt.rcParams["figure.figsize"] = fig_size
plt.legend()
plt.show()

# make a prediction
trainPredict = model.predict(train_X) 
yhat = model.predict(test_X)

# calculate RMSE
trainScore = sqrt(mean_squared_error(y_train, trainPredict))
print('Training of RMSE: %.3f' % trainScore)
RMSE = sqrt(mean_squared_error(y_test, yhat))


#print some rows of predictied values
for x in range(50):
    i = (randint(25, 2000))
    print('Random 50 data')
    print('>> Prediction No. =%d  Actual No. =%.1f' % (int(yhat[i]) ,y_test[i]))
print('>> Prediction No. =%d  Actual No. =%.1f' % (int(yhat[50]) ,y_test[50]))
print('>> Prediction No. =%d  Actual No. =%.1f' % (int(yhat[350]) ,y_test[350]))
print('>> Prediction No. =%d  Actual No. =%.1f' % (int(yhat[216]) ,y_test[216]))
print('>> Prediction No. =%d  Actual No. %.1f' % (int(yhat[1108]) ,y_test[1108]))
print('>> Prediction No. =%d  Actual No. %.1f' % (int(yhat[474]) ,y_test[474]))
print('Test RMSE: %.3f ' % RMSE)

# Validation

print('<<<<<<<<<<<<Validation>>>>>>>>>>>>: \n')

validationSet = pd.read_csv('validation-rnn.csv')

for index, row in validationSet.iterrows():
    array = np.array(row)
    array = array.reshape((1, 1, array.shape[0]))
    
    v_x = array[:, :, 1:]
    v_y = array[:, 0, 0]

    prediction = model.predict(v_x)
    
    error = sqrt(mean_squared_error(v_y, prediction))
    
    print('Predicted No. of people: %.3f' % prediction)
    print('Actual No. of people: ' + repr(int(v_y[0])))
    print('ERROR: %.3f !!!' % error)
    print('\n')

 
