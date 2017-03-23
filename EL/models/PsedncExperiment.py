# -*- coding: utf-8 -*-
#author: luo

from repDNA.psenac import PseDNC
import numpy as np
from numpy import array
from pandas import DataFrame
import time

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


#Pseknc prediction............................................................................
def getCrossValidation(X,y,clf,folds):
    predicted_probability=-np.ones(len(y))
    predicted_label=-np.ones(len(y))
    X=np.array(X)
    y=np.array(y)
    for train_index, test_index in folds:
        X_train = X[train_index]
        X_test  = X[test_index]
        y_train = y[train_index]
        probability_test =(clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index]=probability_test[:, 1]
        predicted_label[test_index]=(clf.fit(X_train, y_train)).predict(X_test)   

    fpr, tpr, thresholds = roc_curve(y, predicted_probability,pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(y,predicted_label)
    sensitivity=recall_score(y,predicted_label)
    specificity=(accuracy*len(y)-sensitivity*sum(y))/(len(y)-sum(y))
    
    return auc_score, accuracy, sensitivity, specificity

###########################################################################################


if __name__ == '__main__':

    featurename='Psednc'

    #getting psednc feature
    print('...............................................................................')
    print('Coding for '+featurename+' feature, beginning')
    tic=time.clock()
        
    psednc = PseDNC(lamada=1, w=0.05)
    pos_vec = psednc.make_psednc_vec(open('posi_samples.fasta'))
    neg_vec = psednc.make_psednc_vec(open('nega_samples.fasta'))
    X = array(pos_vec + neg_vec)
    y=array([1]*len(pos_vec)+[0]*len(neg_vec))
    
    print('The number of positive and negative samples: %d,%d'%(len(pos_vec),len(neg_vec)))
    print('Dimension of '+featurename+' feature vectors: %d'%len(X[0]))
    
    toc=time.clock()
    print("Coding time: %.3f minutes" %((toc-tic)/60.0))
    print('...............................................................................')

    #output the psednc feature    
    np.savetxt(featurename+'Feature.txt',X)
    
    #prediction based on psednc feature
    print('###############################################################################')
    print('The prediction based on '+featurename+', beginning')
    tic=time.clock()

    clf = RandomForestClassifier(random_state=1,n_estimators=500)
    folds = KFold(len(y),n_folds=10,shuffle=True,random_state=np.random.RandomState(1))
    auc_score, accuracy, sensitivity, specificity = getCrossValidation(X, y, clf, folds)
    
    print('results for feature:'+featurename)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****' %(auc_score,accuracy,sensitivity,specificity))
    
    toc=time.clock()
    print('The prediction time: %.3f minutes'%((toc-tic)/60.0))
    print('###############################################################################\n')

    #output result
    results=DataFrame({'Feature':[featurename],\
                      'AUC':[auc_score],\
                      'ACC':[accuracy],\
                      'SN':[sensitivity],\
                      'SP':[specificity]})   
    results=results[['Feature','AUC','ACC','SN','SP']]
    results.to_csv(featurename+'Results.csv',index=False)


