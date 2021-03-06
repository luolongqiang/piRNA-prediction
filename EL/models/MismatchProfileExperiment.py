# -*- coding: utf-8 -*-
#author: luo

import numpy as np
from numpy import array
from pandas import DataFrame
from itertools import combinations_with_replacement, permutations
import time

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


#input sequences................................................................................

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


#getting (k,m)-mismatch profile.................................................................
def getMismatchProfileMatrix(instances, piRNAletter, k, m):
    p=len(piRNAletter)
    kmerdict=getKmerDict(piRNAletter, k)
    features=[]
    if k==1 or k==2:
        for sequence in instances:
            vector=getSpectrumProfileVector(sequence, kmerdict, p, k)
            features.append(vector)
    else:
        if m==1:
            for sequence in instances:
                vector=getMismatchProfileVector(sequence, piRNAletter, kmerdict, p, k)
                features.append(vector)
        else:
            assert('you should reset m<=1')    
    return array(features)
        
def getKmerDict(piRNAletter,k):
    kmerlst=[]
    partkmers=list(combinations_with_replacement(piRNAletter, k))
    for element in partkmers:
        elelst=set(permutations(element, k))
        strlst=[''.join(ele) for ele in elelst]
        kmerlst+=strlst
    kmerlst=np.sort(kmerlst)
    kmerdict={kmerlst[i]:i for i in range(len(kmerlst))}
    return kmerdict


def getSpectrumProfileVector(sequence, kmerdict, p, k):    
    vector=np.zeros((1,p**k))
    n=len(sequence)
    for i in range(n-k+1):
        subsequence=sequence[i:i+k]
        position=kmerdict.get(subsequence)
        vector[0,position]+=1
    return list(vector[0])


def getMismatchProfileVector(sequence, piRNAletter, kmerdict, p, k): 
    n=len(sequence)
    vector=np.zeros((1,p**k))
    for i in range(n-k+1):
        subsequence=sequence[i:i+k]
        position=kmerdict.get(subsequence)
        vector[0,position]+=1
        for j in range(k):
            substitution=subsequence
            for letter in list(set(piRNAletter)^set(subsequence[j])):
                substitution=list(substitution)
                substitution[j]=letter
                substitution=''.join(substitution)
                position=kmerdict.get(substitution)
                vector[0,position]+=1    
    return list(vector[0])


#prediction based on mismatch profile..........................................................
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

    
#############################################################################################

        
if __name__ == '__main__':
    
    featurename='MismatchProfile'
    piRNAletter=['A','C','G','T']

    #input sequences for getting mismatch profile
    fp=open('posi_samples.fasta','r')
    posis=getSequences(fp)
    fn=open('nega_samples.fasta','r')
    negas=getSequences(fn)   
    instances=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))
    print('The number of positive and negative samples: %d, %d'%(len(posis),len(negas)))
    
    #getting (k,m)-mismatch profiles for (k,m)=(1,0),(2,0),(3,1),(4,1),(5,1)
    for args in [[1,0],[2,0],[3,1],[4,1],[5,1]]:
        
        k, m = args[0], args[1]

        print('...............................................................................')
        print('Coding for ('+str(k)+','+str(m)+')-'+featurename+', beginning') 
        tic=time.clock()

        X=getMismatchProfileMatrix(instances, piRNAletter, k, m)
        print('Dimension of ('+str(k)+','+str(m)+')-'+featurename+': %d'%len(X[0]))  
        
        toc=time.clock()
        print('Coding time: %.3f minutes'%((toc-tic)/60.0))
        print('...............................................................................')

        if k==1:
            all_X=X
        else:
            all_X=np.hstack((all_X, X))
        
    #output the mismatch profile
    np.savetxt(featurename+'Feature.txt', all_X)
       
    #prediction based on mismatch profile  
    print('###############################################################################')
    print('The prediction based on '+featurename+', beginning')
    tic=time.clock()

    clf = RandomForestClassifier(random_state=1,n_estimators=500)
    folds = KFold(len(y),n_folds=10,shuffle=True,random_state=np.random.RandomState(1))
    auc_score, accuracy, sensitivity, specificity = getCrossValidation(all_X, y, clf, folds)
  
    print('results for feature:'+featurename)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****' %(auc_score,accuracy,sensitivity,specificity))
    
    toc=time.clock()    
    print('The prediction time: %.3f minutes'%((toc-tic)/60.0))
    print('###############################################################################\n')
    
    #output results
    results=DataFrame({'Feature':[featurename],\
                      'AUC':[auc_score],\
                      'ACC':[accuracy],\
                      'SN':[sensitivity],\
                      'SP':[specificity]})   
    results=results[['Feature','AUC','ACC','SN','SP']]
    results.to_csv(featurename+'Results.csv',index=False)
    