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
import matplotlib.gridspec as gridspec

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


#Pre processing data
#Get labels from far right elements of dataframe
TestLabels = TestDataRaw[TestDataRaw.shape[1]-1]
TrainLabels = TrainDataRaw[TrainDataRaw.shape[1]-1]
TrainData = TrainDataRaw.drop(TrainDataRaw.shape[1]-1,axis=1)
TestData = TestDataRaw.drop(TestDataRaw.shape[1]-1,axis=1)


#Calculating probabilities for each class according to train data
ClassProbability = []
for i in range(0,10):
    ClassProbability.append(len(TrainLabels.loc[TrainLabels==i]))
del i
ClassProbability = pd.DataFrame(ClassProbability)
ClassProbability['probability'] = ClassProbability[0]
ClassProbability = ClassProbability.drop(0, axis=1)
ClassCount = len(ClassProbability)
ClassProbability = ClassProbability/TrainData.shape[0]

# =============================================================================
# INFORMATION GAIN
# =============================================================================

# Entropy using the frequency of classes
#   (✓) In opdigits, it is expected to get something around 3-3.5 of entropy as we have 10 lasses with similar probabilities. Similar probabilities will maximize entropy.
#   (✗) It is expected to get a high (close to 1) gini index, as probabilities are similar
Entropy_N = -np.sum(ClassProbability['probability']*np.log2(ClassProbability['probability'])) 
Gini_N = 1-np.sum(ClassProbability['probability']*ClassProbability['probability'])

#Attribute entropy using the frequency table of attributes:
#   Using pairs of the attributes(64 attributes here) and class
MaxAttributeValue = TrainData.max().max()
MinAttributeValue = 0
i = 1
#For each column/attribute (probably 64 in opdigits data):
#for i in range(0,TrainData.shape[1]): 
TempData_column_and_label =  TrainDataRaw[[i,TrainDataRaw.shape[1]-1]]
AttributeTable = []
AttributeTable_col_sums = []
AttributeTable = pd.DataFrame(AttributeTable)
AttributeTable_col_sums = pd.DataFrame(AttributeTable_col_sums)
#    For each possible value of a column (probably between 0 and 16 in opdigits data):
for j in range(0,MaxAttributeValue+1):
    
#        "attribute & class" table shape / or reverse
#           0 1 2 3 4 5 6 7 8 9 
#       0 \  TempFrequency = [0]  \
#       1 \  TempFrequency = [1]  \
#       2 \  TempFrequency = [2]  \
#       3 \  TempFrequency = [3]  \
#                 ......
#      15 \  TempFrequency = [15] \
#      16 \  TempFrequency = [16] \
     
     ValuesInColumn = TempData_column_and_label.loc[TempData_column_and_label[i]==j] #Rows with value j in column i
 #        Find frequency of classes for the value j of column i (will give a 10 element array):
     TempClassFrequency = []
     for k in range(0,10):
         TempClassFrequency.append(len(ValuesInColumn.loc[ValuesInColumn[TrainDataRaw.shape[1]-1]==k]))
     del k
#         TempClassFrequency = pd.DataFrame(TempClassFrequency)
#         if j>0 :
#             TempClassFrequency[j] = TempClassFrequency[0]
#             TempClassFrequency = TempClassFrequency.drop(0, axis=1)
     AttributeTable[j] = TempClassFrequency
     AttributeTable_col_sums[j] = [np.sum(TempClassFrequency)]
     print(np.sum(TempClassFrequency))
#    Find the entropy of 16 values for each class
     np.sum



class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child, height):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.right = None
        self.left = None
        self.height = None
        
