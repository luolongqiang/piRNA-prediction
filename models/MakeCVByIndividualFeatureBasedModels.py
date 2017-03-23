# -*- coding: utf-8 -*-
#author: longqiang luo

import numpy as np
from numpy import array
from pandas import DataFrame
import time
import sys

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

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
    
def ConstructPartitionOfSet(y, folds_num, seed):
    folds_temp = list(KFold(len(y),n_folds=folds_num,shuffle=True,random_state=np.random.RandomState(seed)))
    folds=[]
    for i in range(folds_num):
        test_index=folds_temp[i][1]
        vali_index=folds_temp[(i+1)%folds_num][1]           
        train_index=array(list(set(folds_temp[i][0])^set(vali_index)))   
        folds.append((train_index, vali_index, test_index))
    return folds

def GetCrossValidation(X, y, feature, clf, folds):
    predicted_probas=-np.ones(len(y))
    predicted_labels=-np.ones(len(y))
    cv_round=1
    for train_index, vali_index, test_index in folds:
        X_train,X_vali,X_test,y_train,y_vali=GetPartitionOfSamples(X,y,feature,train_index,vali_index,test_index)
        predict_test_proba, predict_test_label=MakePrediction(X_train,X_vali,X_test,y_train,y_vali,cv_round)
        predicted_probas[test_index]=predict_test_proba
        predicted_labels[test_index]=predict_test_label 
        cv_round+=1
    auc_score, accuracy, sensitivity, specificity=EvaluatePerformances(y,predicted_probas,predicted_labels)    
    return auc_score, accuracy, sensitivity, specificity
    
def GetPartitionOfSamples(X, y, feature, train_index, vali_index, test_index):
    y_train=y[train_index]
    y_vali =y[vali_index]
    if feature=='Pssm':        
        train_seqs=X[train_index]
        vali_seqs =X[vali_index]
        test_seqs =X[test_index]
        global vdim
        pssm=GetPssmMatrix(train_seqs, y_train, vdim)
        X_train=GetFeatureFromPssm(train_seqs, pssm, vdim)
        X_vali =GetFeatureFromPssm(vali_seqs, pssm, vdim)
        X_test =GetFeatureFromPssm(test_seqs, pssm, vdim)
    else:
        X_train=X[train_index]
        X_vali =X[vali_index]
        X_test =X[test_index]
    return X_train, X_vali, X_test, y_train, y_vali

def MakePrediction(X_train,X_vali,X_test,y_train,y_vali,cv_round):
    classifier=clf.fit(X_train, y_train)
    predict_vali_proba=classifier.predict_proba(X_vali)[:,1]
    fpr, tpr, thresholds = roc_curve(y_vali, predict_vali_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    print('Cross validation,round %d,the AUC is %.3f on validation dataset'%(cv_round,auc_score))    
    predict_test_proba=classifier.predict_proba(X_test)[:,1]
    predict_test_label=classifier.predict(X_test)   
    return predict_test_proba, predict_test_label
    
def EvaluatePerformances(real_labels,predicted_probas,predicted_labels):
    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probas, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(real_labels,predicted_labels)
    sensitivity=recall_score(real_labels,predicted_labels)
    sample_num=len(real_labels)
    posi_num=sum(real_labels)
    specificity=(accuracy*sample_num-sensitivity*posi_num)/(sample_num-posi_num)
    return auc_score, accuracy, sensitivity, specificity

def GetPssmMatrix(train_seqs, y_train, vdim):
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

def GetFeatureFromPssm(seqs, pssm, vdim):  
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
 
########################################################################################
   
if __name__ == '__main__':

    all_features=['1-SpectrumProfile','2-SpectrumProfile','3-SpectrumProfile','4-SpectrumProfile','5-SpectrumProfile',\
                  '(3, 1)-MismatchProfile','(4, 1)-MismatchProfile','(5, 1)-MismatchProfile',\
                  '(3, 1)-SubsequenceProfile','(4, 1)-SubsequenceProfile','(5, 1)-SubsequenceProfile',\
                  '1-RevcKmer','2-RevcKmer','3-RevcKmer','4-RevcKmer','5-RevcKmer',\
                  'PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC','SparseProfile','Pssm']   

    classifier_name=sys.argv[1]
    posi_samples_file=sys.argv[2]
    nega_samples_file=sys.argv[3]
    animal=posi_samples_file.split('_')[0]
    fp=open(posi_samples_file,'r')
    posis=GetSequences(fp)
    fn=open(nega_samples_file,'r')
    negas=GetSequences(fn)   
    instances=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))    
    print('The number of positive and negative samples: %d,%d'%(len(posis),len(negas)))
    
    global vdim 
    vdim=35           #the fixed length of sequences for the PSSM feature  
    folds_num=10      #the number of folds for the cross validation
    seeds_num=1      #the number of seeds for the partition of dataset

    if classifier_name=='RF':
        clf=RandomForestClassifier(random_state=1,n_estimators=500)
    elif classifier_name=='SVM':
        clf=svm.SVC(kernel='rbf',probability=True)
    elif classifier_name=='LR':
        clf=LogisticRegression()

    average_results=0
    for seed in range(1,1+seeds_num):
        print('################################# Seed %d ###################################'%seed)
        start=time.clock()

        folds=ConstructPartitionOfSet(y, folds_num, seed)

        results=[]
        for feature in all_features:
            print('.............................................................................')          
            print('The prediction based on feature:'+feature+', beginning')
            tic=time.clock()

            if feature=='Pssm':
                X=instances
                print('The dimension of the PSSM feature:%d'%vdim)
            else:
                X=np.loadtxt(feature+'Feature'+animal+'.txt') 
                print('The dimension of the '+feature+':%d'%len(X[0]))
            
            auc_score, accuracy, sensitivity, specificity=GetCrossValidation(X, y, feature, clf, folds)
            results.append([auc_score, accuracy, sensitivity, specificity]) 

            toc=time.clock()
            print('*****************************************************************************')  
            print('The final results for feature:'+feature)
            print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****'\
                               %(auc_score, accuracy, sensitivity, specificity))
            print('Running time:%.3f mimutes'%((toc-tic)/60))  
            print('*****************************************************************************')      
            print('.............................................................................\n')    

        results=array(results)
        df=DataFrame({'Feature':all_features,\
                      'AUC':results[:,0],\
                      'ACC':results[:,1],\
                      'SN':results[:,2],\
                      'SP':results[:,3]})
        df=df[['Feature','AUC','ACC','SN','SP']]
        df.to_csv('IndividualFeatureResults'+animal+'CV(seed'+str(seed)+')'+classifier_name+'.csv',index=False)
    
        end=time.clock()
        print('Seed %d, total running time:%.3f minutes'%(seed,(end-start)/60))
        print('#############################################################################')

        average_results+=results

    average_results=average_results/seeds_num
    average_df=DataFrame({'Feature':all_features,\
                          'AUC':average_results[:,0],\
                          'ACC':average_results[:,1],\
                          'SN':average_results[:,2],\
                          'SP':average_results[:,3]})
    average_df=average_df[['Feature','AUC','ACC','SN','SP']]
    average_df.to_csv('IndividualFeatureAverageResults'+animal+'CV('+classifier_name+').csv',index=False)

