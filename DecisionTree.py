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

#Pre processing data
#get labels
TestLabels = TestDataRaw[TestDataRaw.shape[1]-1]
TrainLabels = TrainDataRaw[TrainDataRaw.shape[1]-1]

