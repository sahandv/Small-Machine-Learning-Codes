#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:17:35 2017

@author: sahand
"""

import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
Nodes = {}

# =============================================================================
# Data File Path
# =============================================================================

TestFilePath = r'data/optdigits.tes'
TrainFilePath = r'data/optdigits.tra'

# =============================================================================
# Classes and functions
# =============================================================================
class treeNode():
    def __init__(self, is_leaf, attr_index, attr_table, entropy, parent, height):
        self.is_leaf = is_leaf
        self.attr_split_points = None
        self.attr_index = attr_index
        self.attr_table = attr_table
        self.entropy = entropy
        self.parent = parent
        self.right = None
        self.left = None
        self.height = height
        
def entropy(df_list):
    result = -np.sum(df_list*np.log2(df_list)) 
    return(result)
    
def nodeFinder(TrainData,TrainDataRaw,MaxAttributeValue):
    TotalEntropy = []
    TotalEntropy = pd.DataFrame(TotalEntropy)
    TotalGain = []
    TotalGain = pd.DataFrame(TotalGain)
    AttributeTable_N = {}
    AttributeTable_T = {}
    TrainDataSize = TrainData.shape[0]
    #For each column/attribute (probably 64 in opdigits data):
    for i in range(0,TrainData.shape[1]): 
        TempData_column_and_label =  TrainDataRaw[[i,TrainDataRaw.shape[1]-1]]
        AttributeTable = []
        AttributeTable = pd.DataFrame(AttributeTable)
        AttributeTable_col_sums = []
        AttributeTable_col_sums = pd.DataFrame(AttributeTable_col_sums)
        AttributeTable_probabilities = []
        AttributeTable_probabilities = pd.DataFrame(AttributeTable_probabilities)
        AttributeTable_class_probability = []
        AttributeTable_class_probability = pd.DataFrame(AttributeTable_class_probability)
        AttributeTable_class_entropy = []
        AttributeTable_class_entropy = pd.DataFrame(AttributeTable_class_entropy)
        #    For each possible value of a column (probably between 0 and 16 in opdigits data):
        for j in range(0,MaxAttributeValue+1):
            
        #           AttributeTable.T shape
        #           0 1 2 3 4 5 6 7 8 9 (classes)
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
                 TempClassFrequency.append(len(ValuesInColumn.loc[ValuesInColumn[TrainData.shape[1]]==k]))
             del k
        
             AttributeTable[j] = TempClassFrequency
             AttributeTable_col_sums[j] = [np.sum(TempClassFrequency)]
             AttributeTable_probabilities[j] = [AttributeTable_col_sums[j]/TrainDataSize] #Calculate probability of each value of attribute
        #    Find probability of each class in each value of each column
             AttributeTable_class_probability[j] = TempClassFrequency/np.sum(TempClassFrequency)
        #    Find the entropy of 16 values for each class
             AttributeTable_class_entropy[j] = [entropy(AttributeTable_class_probability[j])]
        
        TotalEntropy[i] = [np.sum(np.sum(AttributeTable_probabilities*AttributeTable_class_entropy))]
        TotalGain[i] = [Entropy_N-TotalEntropy[i]]
        AttributeTable_T[i] = AttributeTable.T
        AttributeTable_N[i] = AttributeTable
    
    #End of loop

    TotalGain = np.max(TotalGain)
    TotalGain = pd.DataFrame(TotalGain)
    TotalGain['gain'] = TotalGain[0]
    TotalGain = TotalGain.drop(0,axis=1)
    MaxGain_index = TotalGain['gain'].argmax() 
    print("\nA node is found with index of "+str(MaxGain_index)+" . This index is the column or attirbute.")
    return(MaxGain_index,AttributeTable_N,AttributeTable_T,TotalEntropy)

# =============================================================================
# INITIALIZATION
# =============================================================================
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
TrainData_temp = TrainData
TestData = TestDataRaw.drop(TestDataRaw.shape[1]-1,axis=1)
TrainDataSize = TrainData.shape[0]
TestDataSize = TestData.shape[0]

#Calculating probabilities for each class according to train data
ClassProbability = []
for i in range(0,10):
    ClassProbability.append(len(TrainLabels.loc[TrainLabels==i]))
del i
ClassProbability = pd.DataFrame(ClassProbability)
ClassProbability['probability'] = ClassProbability[0]
ClassProbability = ClassProbability.drop(0, axis=1)
ClassCount = len(ClassProbability)
ClassProbability = ClassProbability/TrainDataSize

# =============================================================================
# TOTAL / TARGET ENTROPY
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


# =============================================================================
# Tree Generation
# =============================================================================

MaxGain_index,AttributeTable_N,AttributeTable_T,TotalEntropy = nodeFinder(TrainData,TrainDataRaw,MaxAttributeValue)

#Now we have the best break node
#If there is no pure branch, further split is required.
AttributeTable_other_method = TrainDataRaw.pivot_table(0,index=TrainData.shape[1], columns=MaxGain_index,aggfunc='count')

all_is_leaf = False
NodeCount = 0;
counter = 0;
for epoch in range(0,1): 
    print("\n    Constructed Node "+str(counter))
    Nodes[counter] = treeNode(all_is_leaf,MaxGain_index,AttributeTable_T[MaxGain_index],TotalEntropy,None,0)
    counter = counter+1
    
#    if all_is_leaf:
#        TrainData_temp = TrainData_temp.drop(MaxGain_index,axis=1)
#    
#    for i in range(0,MaxAttributeValue+1): 
#        all_is_leaf = False
#        
#        NodeCount=NodeCount+1
#        print("deviding nodes "+str(NodeCount))
#        TrainData_split = TrainData.loc[TrainData[MaxGain_index]==0]
#        TrainDataRaw_split = TrainDataRaw.loc[TrainDataRaw[MaxGain_index]==0]
#        MaxGain_index,AttributeTable_N,AttributeTable_T,TotalEntropy = nodeFinder(TrainData_split,TrainDataRaw_split,MaxAttributeValue)
#        Nodes[counter] = treeNode(all_is_leaf,MaxGain_index,AttributeTable_T[MaxGain_index],TotalEntropy,epoch,1)
#        counter = counter+1
##        
#        NodeCount_kid = 0;
#        for j in range(0,MaxAttributeValue+1):
#            all_is_leaf = False
#            
#            NodeCount_kid=NodeCount_kid+1
#            TrainData_split_kid = TrainData_split.loc[TrainData_split[MaxGain_index]==0]
#            TrainDataRaw_split_kid = TrainDataRaw_split.loc[TrainDataRaw_split[MaxGain_index]==0]
#            print("deviding nodes further "+str(NodeCount_kid))
#            print(TrainData_split_kid.shape[0])
#            MaxGain_index,AttributeTable_N,AttributeTable_T,TotalEntropy = nodeFinder(TrainData_split_kid,TrainDataRaw_split_kid,MaxAttributeValue)
#            Nodes[counter] = treeNode(all_is_leaf,MaxGain_index,AttributeTable_T[MaxGain_index],TotalEntropy,NodeCount,2)
#            counter = counter+1

###############################################################################
#for it in Nodes:
#    print(it,Nodes[it].attr_index)
#    print(Nodes[it].entropy)
#    print(Nodes[it].parent)
#    if Nodes[it].parent<18:
#        print(Nodes[it].attr_index)

#len(Nodes)








































