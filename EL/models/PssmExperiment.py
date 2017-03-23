# -*- coding: utf-8 -*-
#author: luo

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


#input sequences..................................................................................
def getSequences(f):
    seqslst=[]
    while True:
         s=f.readline()
         if not s:
             break
         else:
             if '>' not in s:
                seq=s.split('\n')[0]
                seqslst.append(seq)
    return seqslst


#getting pssm feature..............................................................................
def getPssmMatrix(train_seqs, y_train, vdim):
    alphabet={'A':0,'C':1,'G':2,'T':3}
    posi_train_seqs=train_seqs[list(y_train)]
    posi_train_seqs_num=len(posi_train_seqs)
    alphabet_num=len(alphabet)
    pssm=np.ones((alphabet_num, vdim))*10**(-10)
    for i in range(posi_train_seqs_num):
        seqlen=len(posi_train_seqs[i])
        for j in range(vdim):
            if j<=seqlen-1:
                row_index=alphabet.get(posi_train_seqs[i][j])
                pssm[row_index,j]+=1
    pssm=np.log(pssm*alphabet_num/posi_train_seqs_num)
    return pssm

def getFeatureFromPssm(seqs, pssm, vdim):  
    alphabet={'A':0,'C':1,'G':2,'T':3}
    seqs_num=len(seqs)
    features=np.zeros((seqs_num, vdim))
    for i in range(seqs_num):
        seqlen=len(seqs[i])
        for j in range(vdim):
            if j<=seqlen-1:
                row_index=alphabet.get(seqs[i][j])
                features[i,j]=pssm[row_index,j]    
    return features


#prediction based on the pssm feature.............................................................
def getCrossValidation(instances, y, vdim, clf, folds):
    
    predicted_probability=-np.ones(len(y))
    predicted_label=-np.ones(len(y))
    
    for train_index, test_index in folds:   
        
        train_seqs = instances[train_index]
        test_seqs  = instances[test_index]
        y_train = y[train_index]
        
        pssm=getPssmMatrix(train_seqs, y_train, vdim)
        X_train = getFeatureFromPssm(train_seqs, pssm, vdim)
        X_test = getFeatureFromPssm(test_seqs, pssm, vdim)
        
        probability_test =(clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index]=probability_test[:, 1]
        predicted_label[test_index]=(clf.fit(X_train, y_train)).predict(X_test)   

    fpr, tpr, thresholds = roc_curve(y, predicted_probability,pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(y,predicted_label)
    sensitivity=recall_score(y,predicted_label)
    specificity=(accuracy*len(y)-sensitivity*sum(y))/(len(y)-sum(y))
    
    return auc_score, accuracy, sensitivity, specificity
##############################################################################################

if __name__=='__main__':

    featurename='Pssm'
    vdim=35
    
    #input sequences for getting pssm feature
    fp=open("posi_samples.fasta",'r')
    posis=getSequences(fp)
    fn=open("nega_samples.fasta",'r')
    negas=getSequences(fn)   
    instances=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))
    print('The number of positive and negative samples: %d,%d'%(len(posis),len(negas)))

    #prediction based on pssm feature  
    print('###############################################################################')
    print('The prediction based on '+featurename+' feature, beginning')
    tic=time.clock()
    
    clf=RandomForestClassifier(random_state=1,n_estimators=500)
    folds = KFold(len(y),n_folds=10,shuffle=True,random_state=np.random.RandomState(1))
    auc_score, accuracy, sensitivity, specificity = getCrossValidation(instances, y, vdim, clf, folds)

    print('results for feature:'+featurename)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****' %(auc_score, accuracy, sensitivity, specificity))    
    
    toc=time.clock()
    print('The prediction time: %.3f minutes'%((toc-tic)/60.0))
    print('###############################################################################\n')
    
    #output result
    result=DataFrame({'Feature':[featurename],\
                      'AUC':[auc_score],\
                      'Accuracy':[accuracy],\
                      'Sensitivity':[sensitivity],\
                      'specificity':[specificity]})   
    result=result[['Feature','AUC','Accuracy','Sensitivity','specificity']]
    result.to_csv(featurename+'Results.csv',index=False)
    
    
