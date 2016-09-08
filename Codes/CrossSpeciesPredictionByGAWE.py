# -*- coding: utf-8 -*-
#author: longqiang luo

import numpy as np
from numpy import array
from pandas import DataFrame

import time
import sys

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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

def ModelPrediction(instances,method,threshold):
        
    if method=="GAWE":
        proba_matrix=[]
        label_matrix=[]
        results=[]
        for feature in all_features:
            print('Cross species prediction, results for individual feature-based model: '+feature)
            if feature=="Pssm":
                pssm=np.loadtxt("PssmMatrix"+trained_animal+".txt")
                X=GetFeatureFromPssm(instances, pssm, 35, {'A':0,'C':1,'G':2,'T':3})
            else:
                X=np.loadtxt(feature+"feature"+tested_animal+'.txt')
            classifier=joblib.load(feature+trained_animal+".model")  
            temp_proba=classifier.predict_proba(X)[:,1]
            temp_label=classifier.predict(X)
            auc_score, accuracy, sensitivity, specificity=EvaluatePerformances(y,temp_proba,temp_label)
            results.append([auc_score, accuracy, sensitivity, specificity])
            proba_matrix.append(temp_proba)
            label_matrix.append(temp_label)
        proba_matrix=np.transpose(proba_matrix)
        label_matrix=np.transpose(label_matrix)
        optimal_weights=np.loadtxt("OptimalWeights"+trained_animal+".txt")
        predicted_proba=np.dot(proba_matrix,optimal_weights)
        predicted_label=np.dot(label_matrix,optimal_weights)>threshold 
        SaveResults(results)
    else:  
        if method=="Pssm":
            pssm=np.loadtxt("PssmMatrix"+trained_animal+".txt")
            X=GetFeatureFromPssm(instances, pssm, 35, {'A':0,'C':1,'G':2,'T':3})
        else:
            X=np.loadtxt(feature+"feature"+tested_animal+'.txt')
        classifier=joblib.load(feature+trained_animal+".model")
        predicted_proba=classifier.predict_proba(X)[:,1]
        predicted_label=classifier.predict(X)
    return predicted_proba,predicted_label

def GetPssmMatrix(train_seqs, y_train, vdim, alphabet):
    alphabet_num=len(alphabet)
    alphabet_dict={alphabet[i]:i for i in range(alphabet_num)}
    posi_train_seqs=train_seqs[list(y_train)]
    posi_train_seqs_num=len(posi_train_seqs) 
    pssm=np.ones((alphabet_num, vdim))*10**(-10)
    for i in range(posi_train_seqs_num):
        seqlen=len(posi_train_seqs[i])
        for j in range(vdim):
            if j<=seqlen-1:
                row_index=alphabet_dict.get(posi_train_seqs[i][j])
                pssm[row_index,j]+=1
    pssm=np.log(pssm*alphabet_num/posi_train_seqs_num)
    return pssm, alphabet_dict

def GetFeatureFromPssm(seqs, pssm, vdim, alphabet_dict):  
    seqs_num=len(seqs)
    features=np.zeros((seqs_num, vdim))
    for i in range(seqs_num):
        seqlen=len(seqs[i])
        for j in range(vdim):
            if j<=seqlen-1:
                row_index=alphabet_dict.get(seqs[i][j])
                features[i,j]=pssm[row_index,j]    
    return features

def EvaluatePerformances(real_label,predicted_proba,predicted_label):
    fpr, tpr, thresholds = roc_curve(real_label, predicted_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(real_label,predicted_label)
    sensitivity=recall_score(real_label,predicted_label)
    specificity=(accuracy*len(real_label)-sensitivity*sum(real_label))/(len(real_label)-sum(real_label))    
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****\n'\
                           %(auc_score, accuracy, sensitivity, specificity))

    return auc_score, accuracy, sensitivity, specificity 
    
def SaveResults(results):
    results=array(results)
    df=DataFrame({'Feature':all_features,\
                  'AUC':results[:,0],\
                  'ACC':results[:,1],\
                  'SN':results[:,2],\
                  'SP':results[:,3]})
    df=df[['Feature','AUC','ACC','SN','SP']]
    df.to_csv('IndividualFeatureResults('+trained_animal+'Predict'+tested_animal+').csv',index=False)
    
###############################################################################
    
if __name__ == '__main__':
    
    global all_features
    global trained_animal
    global method
    global tested_animal
    global y

    all_features=['1-SpectrumProfile','2-SpectrumProfile','3-SpectrumProfile','4-SpectrumProfile','5-SpectrumProfile',\
                  '(3, 1)-MismatchProfile','(4, 1)-MismatchProfile','(5, 1)-MismatchProfile',\
                  '(3, 1)-SubsequenceProfile','(4, 1)-SubsequenceProfile','(5, 1)-SubsequenceProfile',\
                  '1-RevcKmer','2-RevcKmer','3-RevcKmer','4-RevcKmer','5-RevcKmer',\
                  'PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC','SparseProfile','Pssm']
     
    trained_animal=sys.argv[1]
    method=sys.argv[2]
    tested_animal=sys.argv[3]
    fp=open(tested_animal+'_posi_samples.fasta','r')
    posis=GetSequences(fp)
    fn=open(tested_animal+'_nega_samples.fasta','r')
    negas=GetSequences(fn)   
    instances=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))

    if len(negas)>len(posis):
        if tested_animal=='Human':
            threshold=0.115
        elif tested_animal=='Drosophila':
            threshold=0.0371
    elif len(negas)==len(posis):
        if tested_animal=='Human':
            threshold=0.441
        elif tested_animal=='Drosophila':
            threshold=0.445
    else:
        threshold=0.5 #default value

    print('*****************************************************************************')
    tic=time.clock() 

    predicted_proba,predicted_label=ModelPrediction(instances,method,threshold)   
    auc_score, accuracy, sensitivity, specificity=EvaluatePerformances(y,predicted_proba,predicted_label)
    df=DataFrame({'AUC':[auc_score], 'ACC':[accuracy], 'SN':[sensitivity], 'SP':[specificity]})
    df=df[['AUC','ACC','SN','SP']]
    df.to_csv("Results("+trained_animal+"Predict"+tested_animal+"By"+method+").csv",index=False)
    
    toc=time.clock()               
    print('Total running time:%.3f minutes'%((toc-tic)/60))
    print('*****************************************************************************\n')

    
        
    
    
    