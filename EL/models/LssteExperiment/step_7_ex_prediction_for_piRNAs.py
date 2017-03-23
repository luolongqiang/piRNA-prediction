# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:37:04 2015

@author: luo
"""

import numpy as np
from numpy import array
import time
from pandas import DataFrame
import string

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def GetFeatures(f):
    features=[]    
    while True:
        s=f.readline()
        if not s:
            break
        elif '>' not in s:
            vector=s.split('\t')
            vector[-1]=vector[-1].split('\n')[0]
            features.append(vector)
    return features
            

def GetCrossValidation(X,y,clf,feature_name):
    folds = KFold(len(y),n_folds=10,shuffle=True,random_state=np.random.RandomState(1))
    predicted_probability=-np.ones(len(y))
    predicted_score=-np.ones(len(y))
    X=array(X)
    y=array(y)
    for train_index, test_index in folds:
        X_train = X[train_index]
        X_test  = X[test_index]
        y_train = y[train_index]
        probability_test =(clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index]=probability_test[:, 1]
        predicted_score[test_index]=(clf.fit(X_train, y_train)).predict(X_test)   

    fpr, tpr, thresholds = roc_curve(y, predicted_probability,pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(y,predicted_score)
    sensitivity=recall_score(y,predicted_score)
    specitivity=(accuracy*len(y)-sensitivity*sum(y))/(len(y)-sum(y))
    
    print('results for feature:'+feature_name)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specitivity:%.3f****'\
                 %(auc_score, accuracy, sensitivity, specitivity))
    return auc_score, accuracy, sensitivity, specitivity
    
#..............................................................................

if __name__ == '__main__':
    
    feature='Lsste'
    
    fp=open('posi_lsste_feature.fasta','r')
    posis=GetFeatures(fp)
    fn=open('nega_lsste_feature.fasta','r')
    negas=GetFeatures(fn)
    X=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))
    print('The number of positive and negative samples: %d, %d'%(len(posis),len(negas)))
    
    print('..............................................................................')
    tic=time.clock()  
    clf=RandomForestClassifier(random_state=1,n_estimators=500)
    auc_score, accuracy, sensitivity, specitivity=GetCrossValidation(X, y, clf, feature)
    toc=time.clock()
    print('runing time %.3f minutes' %((toc-tic)/60))
    print('..............................................................................\n')
    
    result=DataFrame([{'Features':feature,\
                       'AUC':auc_score,\
                       'Accuracy':accuracy,\
                       'Sensitivity':sensitivity,\
                       'Specitivity':specitivity}])
    result=result[['Features','AUC','Accuracy','Sensitivity','Specitivity']]
    result.to_csv(feature+'Results.csv',index=False)
#..............................................................................   
    
    
    

  
