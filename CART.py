# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:14:04 2017

@author: Admin
"""

# Load libraries
import pandas
import numpy as np
#from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "sensor-data.csv"
names = ['Date','Time','Temp', 'CO2', 'Noise','Brightness','Relative-humidity','Motion','No-People','Lights','External-Noise','Internal-Noise','Door-Open','No-Windows','Lecture','No_Computers','Weekday','Vents']
datafile = pandas.read_csv(url)
dataset = pandas.read_csv(url, names=names)
Sensors = datafile.loc[:,'Temp':'Motion']
Time = datafile['Time']
People = datafile['No-People']
SetA =[Time,Sensors]
setA =pandas.concat(SetA,axis=1)

datafile.groupby(Time).describe()
# Create a random dataset
#rng = np.random.RandomState(1)
X = Time#np.sort(5 * rng.rand(80, 1), axis=0)
y = People
#y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()   

#get data groupby time 6-10, 10-15, 15-21, 21-5.9
#if (Time.value>=6 && Time.value<=10):
 #   t=datafile[]
 
 

# dimension
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

#print(datafile[0:5])
# class distribution
#print(dataset.groupby('Time').size(),dataset.groupby('No-People').size(),dataset.groupby('Sensors'))

print(setA[0:5])

# Split-out validation dataset
array = dataset.values
X = array[:,2:19]
Y = array[:,2:8]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)




# Test options and evaluation metric
seed = 7
scoring = 'accuracy'



# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)






## Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()
## box and whisker plots
##dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
##plt.show()
#
#
## histograms
#dataset.hist()
#plt.show()
#
#
## scatter plot matrix
#scatter_matrix(dataset)
#plt.show()
#
#
## scatter plot matrix
#scatter_matrix(dataset)
#plt.show()


#Principle component analysis

