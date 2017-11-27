#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:15:14 2017

@author: sahand
"""

import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier



# =============================================================================
# Classes and functions
# =============================================================================

def confusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    #this func. is taken using stack overflow answers

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print(title)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# compute accuracy per class
def classAccuracy(conf_train, conf_test, flag=False):
	if flag:
		print('Class    train accuracy     test accuracy')
	else:
		print('Class    Before Removal     After Removal')
	for i in range(10):
		train_acc = float(conf_train[i,i])/np.sum(conf_train[i,:])
		test_acc = float(conf_test[i,i])/np.sum(conf_test[i,:])
		print(' {}       {:.4f}           {:.4f}'.format(i, train_acc, test_acc))      


# =============================================================================
# INITIALIZATION
# =============================================================================
TestFilePath = r'data/optdigits.tes'
TrainFilePath = r'data/optdigits.tra'

#Reading file as dataframe and creating sequential column headings
TestDataRaw = pd.read_csv(TestFilePath, sep=',', header=None)
TrainDataRaw = pd.read_csv(TrainFilePath, sep=',', header=None)
del TestFilePath
del TrainFilePath

np.random.seed(seed=300)

#Pre processing data
#Get labels from far right elements of dataframe
TestLabels = TestDataRaw[TestDataRaw.shape[1]-1]
TrainLabels = TrainDataRaw[TrainDataRaw.shape[1]-1]
TrainData = TrainDataRaw.drop(TrainDataRaw.shape[1]-1,axis=1)
TrainData_temp = TrainData
TestData = TestDataRaw.drop(TestDataRaw.shape[1]-1,axis=1)
TrainDataSize = TrainData.shape[0]
TestDataSize = TestData.shape[0]

#Generating 90% train and 10% validation set
TrainData_90, TrainData_validation, TrainLabels_90, TrainLabels_validation = train_test_split(TrainData, TrainLabels, test_size=0.10)

ClassLabels = np.arange(0,10)

# =============================================================================
# KNN
# =============================================================================

kNN_array = np.arange(1,20,2)
AccuracyKNN = []

# Calculating KNN validation
BestNN = None
MaxAcc = -1
for k in kNN_array:
    knn = KNeighborsClassifier(n_neighbors=k) # Constructing "Nearest Neighbor Classification"
    knn.fit(TrainData_90, TrainLabels_90) # Training
    ValidationAccuracy = knn.score(TrainData_validation, TrainLabels_validation) #Computing accuracy
    AccuracyKNN.append(ValidationAccuracy)
    if ValidationAccuracy > MaxAcc:
    	BestNN = knn
    	MaxAcc = ValidationAccuracy

print("Validation accuracy (KNN): {:.4f}".format(BestNN.score(TrainData_validation, TrainLabels_validation)))
BestNeihgbor = kNN_array[AccuracyKNN.index(max(AccuracyKNN))]

# Constructing "Nearest Neighbor Classification"
start = time.time()
knn = KNeighborsClassifier(n_neighbors=BestNeihgbor)
# Training
knn.fit(TrainData, TrainLabels)
print("Nearest neighbors: {}, training time {:.4f} sec.".format(BestNeihgbor, time.time() - start))

# Pridiction using "Nearest Neighbor Classification" lib - train
PredictionsKNN_train = knn.predict(TrainData)
ConfusionKNN_train = confusion_matrix(TrainLabels, PredictionsKNN_train)

# Plot
confusionMatrix(ConfusionKNN_train, classes=ClassLabels, title="KNN Confusion Matrix -Train")

start = time.time()
# Pridiction using "Nearest Neighbor Classification" lib - test
PredictionsKNN_test = knn.predict(TestData)
print("Nearest neighbors: {} testing time {:.4f} sec.".format(BestNeihgbor, time.time() - start))
print("Test accuracy (KNN): {:.4f}\n".format(knn.score(TestData, TestLabels)))
ConfusionKNN_test = confusion_matrix(TestLabels, PredictionsKNN_test)

# Plot
confusionMatrix(ConfusionKNN_test, classes=ClassLabels, title="KNN Confusion Matrix -Test")


# =============================================================================
# Linear Discriminant
# =============================================================================
# Regularization penalties
Regs = [0.0001, 0.001, 0.01, 0.1, 1, 10]
AccuracyLD = []
BestLD = None
MaxAcc = -1
for reg in Regs:
    SGD = linear_model.SGDClassifier(alpha=reg) #Constructing "Linear classifiers" / stochastic gradient descent (SGD)
    SGD.fit(TrainData_90, TrainLabels_90) #Training
    ValidationAccuracy = SGD.score(TrainData_validation, TrainLabels_validation) #Computing accuracy
    AccuracyLD.append(ValidationAccuracy)
    if ValidationAccuracy > MaxAcc:
    	BestLD = SGD
    	MaxAcc = ValidationAccuracy


BestRegulization = Regs[AccuracyLD.index(max(AccuracyLD))]
print ("Best alpha: {}\n".format(BestRegulization))


# Validation accuracy
y_pred_val_sgd = BestLD.predict(TrainData_validation)
print("Validation accuracy (Linear Discriminant): {:.4f}".format(BestLD.score(TrainData_validation, TrainLabels_validation)))

#Constructing "Linear classifiers" / stochastic gradient descent (SGD)
start = time.time()
SGD = linear_model.SGDClassifier(alpha=BestRegulization)
#Training
SGD.fit(TrainData, TrainLabels)
print("alpha: {}, training time {:.4f} sec.".format(BestRegulization, time.time() - start))

# Predicting using "Linear classifiers" lib - train
PredictionsLD_train = SGD.predict(TrainData)
ConfusionSGD_train = confusion_matrix(TrainLabels, PredictionsLD_train)

# Plot
confusionMatrix(ConfusionSGD_train, classes=ClassLabels,title="Linear Classifier Confusion Matrix -Train")


# Predicting using "Linear classifiers" lib - test
start = time.time()
PredictionsLD_test = SGD.predict(TestData)
print("alpha: {}, testing time {:.4f} sec.".format(BestRegulization, time.time() - start))
print("Test accuracy (linear classifier): {:.4f}\n".format(SGD.score(TestData, TestLabels)))
ConfusionSGD_test = confusion_matrix(TestLabels, PredictionsLD_test)

# Plot
confusionMatrix(ConfusionSGD_test, classes=ClassLabels,title="Linear Classifier Confusion Matrix -Test")


# =============================================================================
# Multilayer Perceptron
# =============================================================================

BestMLP = None
MaxAccuracy = -1
BestHL = []
BestRegulization = []
AccuracyMLP = []
HiddenLayers = [64, 128, 256, (64,64), (128,128), (256,256)]
Regs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
for hl in HiddenLayers:
    for reg in Regs:
    	MLP = MLPClassifier(solver='adam', alpha=reg, hidden_layer_sizes=hl)#Constructing 
    	MLP.fit(TrainData_90, TrainLabels_90)#Training
    	ValidationAccuracyMLP = MLP.score(TrainData_validation, TrainLabels_validation) # computing accuracy
    	AccuracyMLP.append(ValidationAccuracyMLP)
    	if ValidationAccuracyMLP > MaxAccuracy:
    		BestMLP = MLP
    		MaxAccuracy = ValidationAccuracyMLP
    		BestHL = hl
    		BestRegulization = reg

print ("Best Hidden Layer Sizes: {}, Best alpha: {}\n".format(BestHL, BestRegulization))

# Validation accuracy
print("Validation accuracy (MLP): {:.4f}".format(BestMLP.score(TrainData_validation, TrainLabels_validation)))

start = time.time()

#Constructing MLP
MLP = MLPClassifier(solver='adam', alpha=BestRegulization, hidden_layer_sizes=BestHL)
MLP.fit(TrainData, TrainLabels)
print("Hiiden Layers: {}, alpha: {}, training time {:.4f} sec.".format(BestHL, BestRegulization, time.time() - start))

# Predicting using "MLP" lib - test
PredictionMLP_train = MLP.predict(TrainData)
ConfusionMLP_train = confusion_matrix(TrainLabels, PredictionMLP_train)
confusionMatrix(ConfusionMLP_train, classes=ClassLabels,title="MLP Confusion Matrix -Train")

start = time.time()
# Predicting using "MLP" lib - test
PredictionMLP_test = MLP.predict(TestData)
print("Hidden Layers: {}, alpha: {}, testing time {:.4f} sec.".format(BestHL, BestRegulization, time.time() - start))
print("Test accuracy (MLP): {:.4f}\n".format(MLP.score(TestData, TestLabels)))
ConfusionMLP_test = confusion_matrix(TestLabels, PredictionMLP_test)

# plot
confusionMatrix(ConfusionMLP_test, classes=ClassLabels,title="MLP Confusion Matrix -Test")



# =============================================================================
# Accuracy of KNN, LD , MLP for all classes
# =============================================================================
print('\n\n # # # # # # # # # # # # # # # # # # # # # # # # # #')
print('     KNN')
print(' # # # # # # # # # # # # # # # # # # # # # # # # # #')

classAccuracy(ConfusionKNN_train, ConfusionKNN_test, flag=True)


print('\n\n # # # # # # # # # # # # # # # # # # # # # # # # # #')
print('     Linear Discriminant Classifier')
print(' # # # # # # # # # # # # # # # # # # # # # # # # # #')
classAccuracy(ConfusionSGD_train, ConfusionSGD_test, flag=True)

print('\n\n # # # # # # # # # # # # # # # # # # # # # # # # # #')
print('     MLP')
print(' # # # # # # # # # # # # # # # # # # # # # # # # # #')
classAccuracy(ConfusionMLP_train, ConfusionMLP_test, flag=True)









































