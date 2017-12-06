# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:25:22 2017

@author: Fati
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#Formatting the plot 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import export_graphviz

dataset = pd.read_csv('sensor-data.csv')

dataset = dataset[['Date','Time','Temp','CO2','Noise', 'Brightness','Relative-humidity','Motion','Nr-People','Weekday']]
dataset['Time'] = dataset['Time'].str.replace(':','.')
dataset['Time'] = dataset['Time'].astype(float)
dataset['Date'] = dataset['Date'].str.replace('/','.')
dataset['Date'] = dataset['Date'].str.replace('.2017 0:00','')
dataset['Date'] = dataset['Date'].astype(float)
dataset.fillna(0, inplace=True)
#Run PCA on First Sensor data
pca = PCA(n_components=2)
#
##Run PCA on Second Sensor data + logsheet
pca = PCA(n_components=2)
pca.fit(dataset)
print("Sensor and logsheet data Variance")
print(pca.explained_variance_ratio_)
print("------------------------------")
print("Sensor and logsheet data Correlation")
print(dataset.corr())
print("------------------------------")
print("Sensor and logsheet data Covariance")
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
             list(df3.columns.values)[i], color='r')

for i in range(len(xs)):
# circles project documents (ie rows from dataset,xlsx sheet) as points onto PC axes
    plt.plot(xs[i], ys[i], 'g')
plt.title('Sensor and logsheet')
plt.rcParams["figure.figsize"] = (70,30)    
plt.show()
