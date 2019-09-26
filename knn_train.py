#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:43:41 2019

@author: anilosmantur
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

n_samples = 30#91

dataNameList = ['attention','meditation','rawValue','delta','theta','lowAlpha','highAlpha',
            'lowBeta','highBeta','lowGamma','midGamma','poorSignal']
featureList = ['attention','meditation','rawValue','delta','theta','lowAlpha','highAlpha',
            'lowBeta','highBeta','lowGamma','midGamma']

labels = ['focus','relax', 'upWord', 'downWord', 
          'upColor', 'downColor', 
          'CyanUP','greenDOWN', 'yellowRIGHT', 'BlackLEFT']#,'blink']

labels = ['relax','upColor','CyanUP']

n_label = len(labels)
#label = labels[2]
#count = 0
trainDataDict = dict()
for data in dataNameList:
    trainDataDict[data] = []
testDataDict = dict()
for data in dataNameList:
    testDataDict[data] = []

def load_data(dataDict, label, count):    
    for data in dataNameList:
        dataDict[data].append(np.load('dataset/{}/{}/{}.npy'.format(label,count,data))[:100])

#n_samples = 10
test_n_samples = int(n_samples/2)
test_size = n_label * int(n_samples/2)
train_n_samples = round(n_samples/2)
train_size = n_label * round(n_samples/2)
#nums = np.arange(n_samples)*2
nums = np.arange(n_samples)
trainNums = np.concatenate([nums[:5],nums[10:15],nums[20:25]])#,nums[31:41], nums[51:61],nums[71:81]])
#trainNums = nums[:5]
np.random.shuffle(trainNums)
testNums = np.concatenate([nums[5:10],nums[15:20],nums[25:30]])#,nums[41:51], nums[61:71],nums[81:91]])
#testNums = nums[5:10]
np.random.shuffle(testNums)
for label in labels:
    for i in trainNums:
        load_data(trainDataDict,label, i)

for label in labels:
    for i in testNums:
        load_data(testDataDict,label, i)


for data in dataNameList:
    trainDataDict[data] = np.array(trainDataDict[data])
for data in dataNameList:
    testDataDict[data] = np.array(testDataDict[data])

#connect features
trainData = []
for data in featureList:
    trainData.append(trainDataDict[data])
trainData = np.array(trainData).transpose(1,0,2)
testData = []
for data in featureList:
    testData.append(testDataDict[data])
testData = np.array(testData).transpose(1,0,2)

trainData = trainData.astype('float32')
testData = testData.astype('float32')
## normalization needed
scaler = MinMaxScaler()
print(scaler.fit(trainData.reshape(-1, 1100)))
trainData = scaler.transform(trainData.reshape(-1, 1100))
testData = scaler.transform(testData.reshape(-1, 1100))

trainLabels = []
for i in range(n_label):
    trainLabels.append(np.ones(train_n_samples)*i )#,np.ones(15)*2])
trainLabels = np.concatenate(trainLabels)

testLabels = []
for i in range(n_label):
    testLabels.append(np.ones(test_n_samples)*i )#,np.ones(15)*2])
testLabels = np.concatenate(testLabels)

from sklearn.model_selection import GridSearchCV

nN = 7
param_grid = {"n_neighbors":np.arange(0,nN)*2 + 1}

print(trainData.reshape(train_size, -1).shape)

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(trainData.reshape(train_size, -1), trainLabels)
print(knn_cv.best_score_)
print(knn_cv.best_params_)

preds = np.array(knn_cv.predict(testData.reshape(test_size, -1)))
probs = np.array(knn_cv.predict_proba(testData.reshape(test_size, -1)))
scores = metrics.accuracy_score(testLabels, preds)
print(' N class: ', n_label)
print('test %: {:6.2f}%'.format(scores*100))

"""
i = 0

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(trainDataDict[dataNameList[i]], Labels)
print(knn_cv.best_score_)
print(knn_cv.best_params_)

"""
nN = 3
neigh = KNeighborsClassifier(n_neighbors=nN)
neigh.fit(trainData.reshape(train_size, -1), trainLabels) 

preds = np.array(neigh.predict(testData.reshape(test_size, -1)))
probs = np.array(neigh.predict_proba(testData.reshape(test_size, -1)))
scores = metrics.accuracy_score(testLabels, preds)
print('N class: ', n_label,'\nn neighbour: ', nN)
print('test %: {:6.2f}%'.format(scores*100))

preds = np.array(neigh.predict(trainData.reshape(train_size, -1)))
probs = np.array(neigh.predict_proba(trainData.reshape(train_size, -1)))
scores = metrics.accuracy_score(trainLabels, preds)
print('N class: ', n_label,'\nn neighbour: ', nN)
print('train %: {:6.2f}%'.format(scores*100))

#import pickle
#
#knnPickle = open('models/knn_best.pkl', 'wb')
#pickle.dump(neigh, knnPickle)

neigh = pickle.load(open('models/knn_best.pkl', 'rb'))
preds = np.array(neigh.predict(testData.reshape(test_size, -1)))
probs = np.array(neigh.predict_proba(testData.reshape(test_size, -1)))
scores = metrics.accuracy_score(testLabels, preds)
print('N class: ', n_label,'\nn neighbour: ', nN)
print('test %: {:6.2f}%'.format(scores*100))

preds = np.array(neigh.predict(trainData.reshape(train_size, -1)))
probs = np.array(neigh.predict_proba(trainData.reshape(train_size, -1)))
scores = metrics.accuracy_score(trainLabels, preds)
print('N class: ', n_label,'\nn neighbour: ', nN)
print('train %: {:6.2f}%'.format(scores*100))