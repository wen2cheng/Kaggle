# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:15:32 2017

@author: wencheng
"""

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import re

def get_title(name):
    # 正则表达式检索称谓，称谓总以大写字母开头并以句点结尾
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果称谓存在则返回其值
    if title_search:
        return title_search.group(1)
    return ""


titanic = pd.read_csv(r"train.csv")

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
           
           
# 创建一个新的Series对象titles，统计各个头衔出现的频次
titles = titanic["Name"].apply(get_title)
#print(pd.value_counts(titles))

# 将每个称谓映射到一个整数，有些太少见的称谓可以压缩到一个数值
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# 验证转换结果
#print(pd.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles          
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))           
           
           
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize","NameLength"]        

# 测试数据

titanic_test = pd.read_csv(r"test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# 将male替换为1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1

# 将female替换为0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0


# 首先把所有缺失值替换为"S"
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

# 将"S"替换为0
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
#将“C”替换为1
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
#将“C”替换为2
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2  
titanic_test["Title"] = titles  
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"] 
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
                
# 初始化逻辑回归类
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

## 训练算法
alg.fit(titanic[predictors], titanic["Survived"])
#计算交叉验证得分
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print("RandomForestClf估计的准确率为%f"%scores.mean())


# 对测试集做预测
predictions = alg.predict(titanic_test[predictors])

# 创建新的dataframe对象submission，仅含"PassengerID"和"Survived"两列。
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

# 将submission输出为.csv格式的提交文件
submission.to_csv("kaggle.csv", index=False)                
  


             
##SVM
clf = SVC(C=0.5)
clf.fit(titanic[predictors], titanic["Survived"])
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(clf, titanic[predictors], titanic["Survived"], cv=kf)
print("SVM估计的准确率为%f"%scores.mean())         
   

# NN
clf1 = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='logistic', solver='adam',
                    learning_rate_init = 0.01, max_iter=3000) 
clf1.fit(titanic[predictors], titanic["Survived"])
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(clf1, titanic[predictors], titanic["Survived"], cv=kf)
print("mlp估计的准确率为%f"%scores.mean())            

#
modle = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')              
                
modle.fit(titanic[predictors],titanic["Survived"])
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(modle, titanic[predictors], titanic["Survived"], cv=kf)
print("DT估计的准确率为%f"%scores.mean())               
        

#GBDT
gb_clf = GradientBoostingClassifier( n_estimators=50, max_depth=3,random_state=1)
gb_clf.fit(titanic[predictors],titanic["Survived"])
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(gb_clf, titanic[predictors], titanic["Survived"], cv=kf)
print("GBDT估计的准确率为%f"%scores.mean())  

# KNN
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=10)
knn.fit(titanic[predictors],titanic["Survived"])
kf = cross_validation.KFold(titanic.shape[0], 5, random_state=1)
scores = cross_validation.cross_val_score(knn, titanic[predictors], titanic["Survived"], cv=kf)
print("KNN估计的准确率为%f"%scores.mean()) 



##ensemble              
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked","Title"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "Age", "Embarked","Title"]],
    [neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=5),["Pclass", "Sex", "Fare", "Age", "Embarked","Title"]],
    [clf1,["Pclass", "Sex", "Fare", "Age", "Embarked","Title"]],
]

# 初始化交叉验证
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # 对每个交叉验证分组，分别使用两种算法进行分类
    for alg, predictors in algorithms:
        # 用训练集拟合算法
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # 选择并预测测试集上的输出 
        # .astype(float) 可以把dataframe转换为浮点数类型
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # 对两个预测结果取平均值
    test_predictions = (full_test_predictions[0]*5 + full_test_predictions[1]) / 6
    # 大于0.5的映射为1；小于或等于的映射为0
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# 将预测结果存入一个数组
predictions = np.concatenate(predictions, axis=0)

# 与训练集比较以计算精度
accuracy = len(predictions[predictions == titanic["Survived"]]) / len(predictions)

print("组合提升估计的准确率为%f"%accuracy)            
# 创建新的dataframe对象submission，仅含"PassengerID"和"Survived"两列。
#submission = pd.DataFrame({
#        "PassengerId": titanic_test["PassengerId"],
#        "Survived":    predictions
#    })
## 将submission输出为.csv格式的提交文件
#submission.to_csv("kaggle_combine.csv", index=False)  





















        