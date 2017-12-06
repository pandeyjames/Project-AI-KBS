# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:59:16 2017

@author: Admin
""" 
#https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/


import math
from pandas import read_csv 
import random
import operator
from sklearn.metrics import mean_squared_error

#first we have the get data from the set of data
def getDataset(filename, split, trainingSet=[], testSet=[] ):
    data = read_csv(filename)
    data.fillna(0,inplace=True)
    df=data[['Time','Noise','Brightness','Relative-humidity','Motion','Nr-People']] 
    df['Time'] = df['Time'].str.replace(':','.')
    df['Time'] = df['Time'].astype(float)
    

    lines = df.values
    dataset = list(lines)
    for x in range( len(dataset)-1 ):
        if random.random() < split:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])


#To calculate Eucleidean Distance
def getEuclideanDistance(var1, var2, leng):  
    dist = 0
    for x in range(leng):
        x1 = float(var1[x])
        x2 = float(var2[x])
        dist += pow( (x1 - x2), 2)
    return math.sqrt(dist)
    

# training, testset,  k = distance
def getNeighbours(trainingset, testset, k):
    leng = len(testset) -1
    dist = []
    for x in range(len(trainingset)):
        d = getEuclideanDistance(testset, trainingset[x], leng)
        dist.append((trainingset[x], d))
    dist.sort(key=operator.itemgetter(1))
    neighbours = []
    
    # For the x (number of closest) in k (specified distance) aka Find me the K nearest neighbours
    for x in range(k):
        neighbours.append(dist[x][0])
    return neighbours

def getResponse(neighbours):
    vote = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in vote:
            vote[response] += 1
        else:
            vote[response] = 1
    sortvote = sorted(vote.items(), key=operator.itemgetter(1), reverse = True)
    return sortvote[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        test = int(testSet[x][-1])
        pred = int(predictions[x])
        
        if test is pred:
            correct += 1
        
    return (correct / float(len(testSet))) * 100.0

def getValidation(filename):
    vSet = read_csv(filename)
    vSet.fillna(0, inplace=True)
    validation = vSet.values
    
    return validation
    

def main():
    # prepare data
    trainingset = []
    testset = []
    split = 0.66 
    getDataset('sensor-data.csv', split, trainingset, testset)
    

    # Generate predictions
    predictions = []
    k = 7
    for x in range(len(testset)):
        neighbours = getNeighbours(trainingset, testset[x], k)
        result = getResponse(neighbours)
        predictions.append(result)
        print('> prediction=' + repr(result) + ', true data=' + repr(testset[x][-1]) + '\n')
    accuracy = getAccuracy(testset, predictions)
    print('Accuracy for test set: ' + repr(accuracy) + '%')
    
    print('Now validating \n')
    
    # Validation
    validationPredictions = []
    validationSet = getValidation('validation-knn.csv')
    for x in range(len(validationSet)):
        v_neighbours = getNeighbours(trainingset, validationSet[x], k)
        v_result = getResponse(v_neighbours)
        validationPredictions.append(v_result)
        print('> predicted number=' + repr(v_result) + ', actual number=' + repr(validationSet[x][-1]) + '\n')
    v_accuracy = getAccuracy(validationSet, validationPredictions)
    print('Accuracy for validation set: ' + repr(v_accuracy) + '%')



main()
