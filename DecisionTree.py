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


TestFilePath = r'data/optdigits.tes'
TrainFilePath = r'data/optdigits.tra'

TestDataRaw = pd.read_csv(TestFilePath, sep=',', header=None)
TrainDataRaw = pd.read_csv(TrainFilePath, sep=',', header=None)

del TestFilePath
del TrainFilePath

#Pre processing data
#Get labels
TestLabels = TestDataRaw[TestDataRaw.shape[1]-1]
TrainLabels = TrainDataRaw[TrainDataRaw.shape[1]-1]

#Class Probabilities
ClassProbability = []
for i in range(0,10):
    ClassProbability.append(len(TrainLabels.loc[TrainLabels==i]))
del i




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
        
