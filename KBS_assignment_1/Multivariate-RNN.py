# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:14:04 2017

@author: Admin
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot #to plot graph
 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
dataset = pd.read_csv('sensor_data.csv', header=0, index_col = 0, delimiter=";")
dataset.fillna(0, inplace=True)

# Nr-people have to be First
#df = dataset[['Nr-People', 'Motion', 'Brightness', 'Nr-Computers']] # Mixed data, use validation2.csv
df = dataset[['Nr-People', 'Time', 'Noise', 'Brightness', 'Motion']] # Monitor data, validation.csv
values = df.values

# Ensure all data is float
values = values.astype('float32')

x = values[:, 1:]   # Everything but the first column, 
y = values[:, 0]    # First column only, the Answers


train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42)

# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


## Design network
timesteps = train_X.shape[1]
data_dim = train_X.shape[2]

model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, data_dim)))
model.add(Dense(1, activation='relu'))
print(model.summary())
model.compile(loss='mae', optimizer='adam')

# Fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Plot history
pyplot.rcParams['figure.figsize'] = [5, 3]
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction (yhat = predictions)
yhat = model.predict(test_X)

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
print('\n')

# Validation

print('Testing specific feature vectors: \n')

validationSet = pd.read_csv('validation.csv', header=0, index_col = 0, delimiter=";")

for index, row in validationSet.iterrows():
    array = np.array(row)
    array = array.reshape((1, 1, array.shape[0]))
    
    v_x = array[:, :, 1:]
    v_y = array[:, 0, 0]

    prediction = model.predict(v_x)
    
    error = sqrt(mean_squared_error(v_y, prediction))
    
    print('Predicted number of people in the room: %.3f' % prediction)
    print('Actual number of people in the room: ' + repr(int(v_y[0])))
    print('The error for this prediction was: %.3f' % error)
    print('\n')



