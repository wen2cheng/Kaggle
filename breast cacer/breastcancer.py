# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 22:34:38 2017

@author: wencheng
"""

import tensorflow as tf
import numpy as np
import pandas as pd

#准备数据
train = pd.read_csv("breast-cancer-train.csv")
test = pd.read_csv("breast-cancer-test.csv")

#分隔特征与标签
x_train = np.float32(train[['Clump Thickness','Cell Size']].T)
y_train = np.float32(train['Type'].T)

x_test = np.float32(test[['Clump Thickness','Cell Size']].T)
y_test = np.float32(test['Type'].T)

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(w , x_train)+b

loss = tf.reduce_mean(tf.square(y-y_train))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for step in range(0,1000):
    sess.run(train)
    if step%200 == 0:
        print(step ,sess.run(w) ,sess.run(b))
#准备测试样本
test_negative = test.loc[test['Type']==0][['Clump Thickness','Cell Size']]
test_positive = test.loc[test['Type']==1][['Clump Thickness','Cell Size']]        

#作图
import matplotlib.pyplot as plt

plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

lx = np.arange(0,12)

plt.show()










            