#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:15:14 2017

@author: sahand
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


test_file_address = r'optdigits.tes'
train_file_address = r'optdigits.tra'

test_data_raw = pd.read_csv(test_file_address, sep=',')
train_data_raw = pd.read_csv(train_file_address, sep=',')




