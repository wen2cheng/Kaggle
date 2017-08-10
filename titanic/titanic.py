# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:15:32 2017

@author: wencheng
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
import operator

titanic = pd.read_csv(r"C:\Users\wencheng\Desktop\titanic\train.csv")

titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
#print(titanic.describe())
# 确认所有不重复值——应该只有male/female
print(titanic["Sex"].unique())

# 将male替换为1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1

# 将female替换为0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

# 输出"Embarked"的所有数据
print(titanic["Embarked"].unique())

# 首先把所有缺失值替换为"S"
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# 将"S"替换为0
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
#将“C”替换为1
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
#将“C”替换为2
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2       
           
           
           
           