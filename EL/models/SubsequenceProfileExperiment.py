# -*- coding: utf-8 -*-
#author: luo

import numpy as np
from numpy import array
from pandas import DataFrame
from itertools import combinations, combinations_with_replacement, permutations
import time
import multiprocessing

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#input sequences........................................................................
def GetSequences(f):
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


#getting (k, delta)-subsequence profile...................................................
def GetSubsequenceProfileByParallel(instances, piRNAletter, k, delta):
    cpu_num=multiprocessing.cpu_count()   
    batches=ConstructPartitions(instances, cpu_num)
    pool = multiprocessing.Pool(cpu_num)
    results=[]
    for batch in batches:
        temp=pool.apply_async(GetSubsequenceProfile, (batch, piRNAletter, k, delta))
        results.append(temp)
    pool.close()
    pool.join()
    i=1
    for temp in results:
        temp_X=temp.get()
        if i==1:
            X=temp_X
        else:
            X=np.vstack((X,temp_X))
        i+=1
    return X
    
def ConstructPartitions(instances, cpu_num):
    seqs_num=len(instances)
    batch_num=seqs_num//cpu_num
    batches=[]
    for i in range(cpu_num-1):
        batch=instances[i*batch_num:(i+1)*batch_num]
        batches.append(batch)
    batch=instances[(cpu_num-1)*batch_num:]
    batches.append(batch)
    return batches
    
def GetSubsequenceProfile(instances, piRNAletter, k, delta):
    kmerdict=GetKmerDict(piRNAletter, k)
    X=[]
    for sequence in instances:
        vector=GetSubsequenceProfileVector(sequence, kmerdict, k, delta)
        X.append(vector)
    X=array(X)    
    return X

def GetSubsequenceProfileVector(sequence, kmerdict, k, delta):      
    vector=np.zeros((1,len(kmerdict)))
    sequence=array(list(sequence))
    n=len(sequence)
    index_lst=list(combinations(range(n), k))
    for subseq_index in index_lst:
        subseq_index=list(subseq_index)
        subsequence=sequence[subseq_index]
        position=kmerdict.get(''.join(subsequence))     
        subseq_length=subseq_index[-1] - subseq_index[0] + 1
        subseq_score=1 if subseq_length==k else delta**subseq_length    
        vector[0,position]+=subseq_score
    return list(vector[0])

def GetKmerDict(piRNAletter,k):
    kmerlst=[]
    partkmers=list(combinations_with_replacement(piRNAletter, k))
    for element in partkmers:
        elelst=set(permutations(element, k))
        strlst=[''.join(ele) for ele in elelst]
        kmerlst+=strlst
    kmerlst=np.sort(kmerlst)
    kmerdict={kmerlst[i]:i for i in range(len(kmerlst))}
    return kmerdict 


#prediction based on subsequence profile...............................................
def GetCrossValidation(X,y,clf,folds):
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
    
#########################################################################################


if __name__=='__main__':

    featurename='SubsequenceProfile'
    piRNAletter=['A','C','G','T']
     
    #input sequences
    fp=open("posi_samples.fasta",'r')
    posis=GetSequences(fp)
    fn=open("nega_samples.fasta",'r')
    negas=GetSequences(fn)   
    instances=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))
    print('The number of positive and negative samples: %d, %d'%(len(posis),len(negas)))	
    
    #getting (k,delta)-subsequence profiles for (k,delta)=(1,0),(2,0),(3,1),(4,1),(5,1)
    print('This process of getting features may spend some time \nplease do not close the program')
    for args in [[1,0],[2,0],[3,1],[4,1],[5,1]]:

        k, delta = args[0], args[1] 

        print('...............................................................................')
        print('Coding for ('+str(k)+','+str(delta)+')-'+featurename+', beginning') 
        tic=time.clock()

        X=GetSubsequenceProfileByParallel(instances, piRNAletter, k, delta)
        print('Dimension of ('+str(k)+','+str(delta)+')-'+featurename+': %d'%len(X[0]))  

        toc=time.clock()
        print('Coding time: %.3f minutes'%((toc-tic)/60.0))
        print('...............................................................................')

        if k==1:
            all_X=X
        else:
            all_X=np.hstack((all_X, X))

    #output the subsequence profile
    np.savetxt(featurename+'Feature.txt', all_X)

    #prediction based on subsequence profile  
    print('###############################################################################')
    print('This process of prediction may spend several minutes, please do not close the program')
    print('The prediction based on '+featurename+', beginning')
    tic=time.clock()
   
    clf = RandomForestClassifier(random_state=1,n_estimators=500)
    folds = KFold(len(y),n_folds=10,shuffle=True,random_state=np.random.RandomState(1))
    auc_score, accuracy, sensitivity, specificity = GetCrossValidation(all_X, y, clf, folds)

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
    