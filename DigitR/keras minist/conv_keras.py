# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:06:55 2017

@author: wencheng
"""

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.reshape((len(X_train),1,28,28))
y_train = to_categorical(y_train,num_classes=10)

X_test=X_test.reshape((len(X_test),1,28,28))
y_test = to_categorical(y_test,num_classes=10)

from keras.layers import Dense,Dropout,concatenate
from keras.models import Model
import keras

input_img = Input(shape=(1,28,28))
tower_1 = Conv2D(32,(5,5),padding='same',activation='relu')(input_img)
tower_1 = MaxPooling2D((2,2),padding='same',strides=(1,1))(tower_1)


tower_2 = Conv2D(64,(5,5),padding='same',activation='relu')(tower_1)
tower_2 = MaxPooling2D((2,2),padding='same',strides=(1,1))(tower_2)

#tower_2 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
#tower_2 = Conv2D(64,(5,5),padding='same',activation='relu')(tower_2)
#
#tower_3 = MaxPooling2D((3,3),padding='same',strides=(1,1))(input_img)
#tower_3 = Conv2D(64,(5,5),padding='same',activation='relu')(tower_3)
#
#output = concatenate([tower_1,tower_2,tower_3],axis=1)

output = keras.layers.core.Flatten()(tower_2)

res = Dense(300,activation='sigmoid')(output)
#res =Dropout(0.5)
pre = Dense(10,activation='softmax')(res)

model = Model(inputs=input_img,outputs=pre)



#input_tensor = Input(output.shape[1:])
#x = Dropout(0.5)(input_tensor)
#x = Dense(64, activation='sigmoid')(x)
#pre = Dense(10,activation='softmax')(x)
#model = Model(input_tensor, x)
#model = Model(inputs=input_tensor,outputs=output)


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=128, epochs=10, validation_split=0.2)