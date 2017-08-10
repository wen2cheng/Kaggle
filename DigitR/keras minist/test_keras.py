# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:55:58 2017

@author: wencheng
"""

from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
y_train = to_categorical(y_train,num_classes=10)
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
y_test = to_categorical(y_test,num_classes=10)

from keras.layers import Input,Dense,Dropout
from keras.models import Model

inputs = Input(shape=(784,))

x= Dense(1000,activation='sigmoid')(inputs)
x= Dropout(0.5)(x)
x= Dense(64,activation='sigmoid')(x)
pre = Dense(10,activation='softmax')(x)

model = Model(inputs = inputs,outputs=pre)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=128, epochs=10, validation_split=0.2)
