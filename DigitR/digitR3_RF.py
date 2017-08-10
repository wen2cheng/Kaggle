# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:58:03 2017

@author: wencheng
"""

from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
from sklearn.cross_validation import cross_val_score
import pandas as pd

start = time.clock()


print("reading data")
dataset = pd.read_csv(r'train.csv')
X_train = dataset.values[0:,1:]
Y_train = dataset.values[0:,0]

## for small evaluation
#from sklearn.grid_search import GridSearchCV
#x_train_small = X_train[:10000,:]
#y_train_small = Y_train[:10000]
#
#elapsed = (time.clock()-start)
#
##processing
#parameters = {'criterion':['gini','entropy'],'max_features':['auto',12,100]}
#
#rf_clf = RandomForestClassifier(n_estimators=400,n_jobs=4,verbose=1)
#gs_clf = GridSearchCV(rf_clf,parameters,n_jobs=1,verbose=True)
#
#gs_clf.fit(x_train_small.astype('int'),y_train_small)
#
#print()
#for params,mean_score,scores in gs_clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"  % (mean_score, scores.std() * 2, params))
#print()
##end time
#elapsed = (time.clock() - start)


##print("Time used:",elapsed) #seconds
##0.945 (+/-0.004) for {'criterion': 'gini', 'max_features': 'auto'}
##0.946 (+/-0.007) for {'criterion': 'gini', 'max_features': 12}
##0.943 (+/-0.009) for {'criterion': 'gini', 'max_features': 100}
##0.944 (+/-0.003) for {'criterion': 'entropy', 'max_features': 'auto'}
##0.945 (+/-0.004) for {'criterion': 'entropy', 'max_features': 12}
##0.941 (+/-0.003) for {'criterion': 'entropy', 'max_features': 100}

clf = RandomForestClassifier(n_estimators = 12)
clf.fit(X_train,Y_train)

X_test = pd.read_csv(r'test.csv').values
result = clf.predict(X_test)
result=np.c_[range(1,len(result)+1),result.astype(int)] #转化为int格式生成一列
df_result = pd.DataFrame(result,columns = ['ImagsID','Label'])
df_result.to_csv('result.rf.csv',index = False)

score = cross_val_score(clf, X_train, Y_train, cv=3)
print( score.mean() )
















