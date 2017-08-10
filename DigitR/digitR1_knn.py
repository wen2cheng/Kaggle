# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 23:23:16 2017

@author: wencheng
"""

import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier



#read data
print ("reading data")
dataset = pd.read_csv(r"C:\Users\wencheng\Desktop\DigitR/train.csv")
X_train = dataset.values[0:, 1:]
y_train = dataset.values[0:, 0]


#for fast evaluation
X_train_small = X_train[:10000, :]
y_train_small = y_train[:10000]


X_test = pd.read_csv(r"C:\Users\wencheng\Desktop\DigitR/test.csv").values


#knn
#-----------------------用于小范围测试选择最佳参数-----------------------------#
#begin time
start = time.clock()

#progressing
print("selecting best paramater range")
knn_clf=KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, X_train_small, y_train_small, cv=3)

print( score.mean() )
#end time
elapsed = (time.clock() - start)
print("Time used:",int(elapsed), "s")
