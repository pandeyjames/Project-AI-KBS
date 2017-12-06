# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:22:02 2017

@author: Admin
"""

import pandas as pd
#Formatting the plot 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


dataset = pd.read_csv('sensor-data.csv')

dataset = dataset[['Date','Time','Temp','CO2','Noise', 'Brightness','Relative-humidity','Motion','Nr-People','Weekday']]
dataset['Time'] = dataset['Time'].str.replace(':','.')
dataset['Time'] = dataset['Time'].astype(float)
#dataset['Date'] = dataset['Date'].str.replace('-','.')
#dataset['Date'] = dataset['Date'].str.replace('.17 0:00','')
#dataset['Date'] = dataset['Date'].astype(float)
dataset = dataset.drop('Date', 1)
dataset.fillna(0, inplace=True)
#Run PCA on First Sensor data
pca = PCA(n_components=2)
#
##Run PCA on Second Sensor data + logsheet
pca = PCA(n_components=2)
pca.fit(dataset)
print("Sensor and Logsheet Data Variance")
print(pca.explained_variance_ratio_)
print("<<<<<<<<<------------------------------>>>>>>>>>>>>>>>")
print("Sensor and Logsheet Data Co-rrelation")
print(dataset.corr())
print("<<<<<<<<<<<<------------------------------>>>>>>>>>>>>")
print("Sensor and Logsheet Data Co-variance")
print(dataset.cov())
#Lodaing plot
xvector = pca.components_[0]
yvector = pca.components_[1]
xs = pca.transform(dataset)[:,0]
ys = pca.transform(dataset)[:,1]

for i in range(len(xvector)):
# arrows project features (ie columns from dataset,xlsx sheet) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(dataset.columns.values)[i], color='r')

for i in range(len(xs)):
# circles project documents (ie rows from dataset,xlsx sheet) as points onto PC axes
    plt.plot(xs[i], ys[i], 'g')
plt.title('Sensor and Logsheett')
plt.rcParams["figure.figsize"] = (70,30)    
plt.show()
