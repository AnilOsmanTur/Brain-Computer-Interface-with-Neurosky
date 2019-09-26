#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:29:28 2019

@author: anilosmantur
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

np.random.seed(42)

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


trainDataDict = dict()
for data in dataNameList:
    trainDataDict[data] = []
testDataDict = dict()
for data in dataNameList:
    testDataDict[data] = []
valDataDict = dict()
for data in dataNameList:
    valDataDict[data] = []

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

valNums = testNums[:int(len(testNums)/2)]
testNums = testNums[int(len(testNums)/2):]

for label in labels:
    for i in trainNums:
        load_data(trainDataDict,label, i)

for label in labels:
    for i in testNums:
        load_data(testDataDict,label, i)

for label in labels:
    for i in valNums:
        load_data(valDataDict,label, i)
        

for data in dataNameList:
    trainDataDict[data] = np.array(trainDataDict[data])
for data in dataNameList:
    testDataDict[data] = np.array(testDataDict[data])
for data in dataNameList:
    valDataDict[data] = np.array(valDataDict[data])

#connect features
trainData = []
for data in featureList:
    trainData.append(trainDataDict[data])
trainData = np.array(trainData).transpose(1,0,2)

testData = []
for data in featureList:
    testData.append(testDataDict[data])
testData = np.array(testData).transpose(1,0,2)

valData = []
for data in featureList:
    valData.append(valDataDict[data])
valData = np.array(valData).transpose(1,0,2)


trainLabels = []
for i in range(n_label):
    trainLabels.append(np.ones(int(n_samples/2))*i )#,np.ones(15)*2])
trainLabels = np.concatenate(trainLabels)
train_indexes = np.arange(len(trainLabels))
np.random.shuffle(train_indexes)
valLabels = []
for i in range(n_label):
    valLabels.append(np.ones(len(valNums))*i )#,np.ones(15)*2])
valLabels = np.concatenate(valLabels)
val_indexes = np.arange(len(valLabels))
np.random.shuffle(val_indexes)
testLabels = []
for i in range(n_label):
    testLabels.append(np.ones(len(testNums))*i )#,np.ones(15)*2])
testLabels = np.concatenate(testLabels)
test_indexes = np.arange(len(testLabels))
np.random.shuffle(test_indexes)


x_train = trainData[train_indexes]
x_val = valData[val_indexes]
x_test = testData[test_indexes]

y_train = trainLabels[train_indexes]
y_val = valLabels[val_indexes]
y_test = testLabels[test_indexes]

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K


batch_size = 4
num_classes = n_label
epochs = 50

# input image dimensions
img_rows, img_cols = 10, 10
channel = 11
# the data, split between train and test sets

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
## normalization needed
scaler = MinMaxScaler()
print(scaler.fit(x_train.reshape(-1, 1100)))
x_train = scaler.transform(x_train.reshape(-1, 1100))
x_test = scaler.transform(x_test.reshape(-1, 1100))
x_val = scaler.transform(x_val.reshape(-1, 1100))

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channel, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channel, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], channel, img_rows, img_cols)
    input_shape = (channel, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channel)
    input_shape = (img_rows, img_cols, channel)




print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_val.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

checkp = ModelCheckpoint('models/best_model.hdf5', monitor='val_loss', save_best_only=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[checkp])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {:.3f}'.format(score[0]))
print('Test accuracy: {:6.3f}%'.format(score[1]*100))

model.load_weights('models/best_model.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {:.3f}'.format(score[0]))
print('Test accuracy: {:6.3f}%'.format(score[1]*100))