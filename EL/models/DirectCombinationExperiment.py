# -*- coding: utf-8 -*-
#author: luo

import numpy as np
from numpy import array
from pandas import DataFrame
import itertools
import time

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#input sequences.........................................................................
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

#global cross validation:10-fold.........................................................
def globalCrossValidation(all_X, global_y, all_features, feature_combinations, classifier, folds):  
    predicted_probability=-np.ones(len(global_y))
    predicted_label=-np.ones(len(global_y))
    all_optimal_feature_combinations=[]
    cv_round=1
    for train_index, test_index in folds: 
        print('..........................................................................')
        print('global cross validation, round %d, beginning'%cv_round)
        start=time.clock()
               
        optimal_combination = GetOptimalFeatureCombination(all_X, global_y, train_index, all_features, feature_combinations)                
        optimal_X_train, optimal_X_test, global_y_train = ConstructTrainAndTestDataset(all_X, global_y, optimal_combination, train_index, test_index)
        
        for i in range(len(optimal_combination)):
            if i==0:
                optimal_merged_feature_name=all_features[optimal_combination[i]]
            else:
                optimal_merged_feature_name+='+'+all_features[optimal_combination[i]]
        all_optimal_feature_combinations.append(optimal_merged_feature_name)
        
        probability_temp =(classifier.fit(optimal_X_train, global_y_train)).predict_proba(optimal_X_test)
        predicted_probability[test_index]=probability_temp[:, 1]
        predicted_label[test_index]=(classifier.fit(optimal_X_train, global_y_train)).predict(optimal_X_test)        
        
        end=time.clock()
        print('round '+str(cv_round)+', optimal feature combination: '+optimal_merged_feature_name)
        print('round %d, running time: %.3f hour'%(cv_round, (end-start)/3600))
        print('..........................................................................\n')        
        cv_round+=1
        
    unique_optimal_feature_combination={element:all_optimal_feature_combinations.count(element) for element in all_optimal_feature_combinations}
     
    fpr, tpr, thresholds = roc_curve(global_y, predicted_probability, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(global_y,predicted_label)
    sensitivity=recall_score(global_y,predicted_label)
    specificity=(accuracy*len(global_y)-sensitivity*sum(global_y))/(len(global_y)-sum(global_y))
  
    return auc_score, accuracy, sensitivity, specificity, unique_optimal_feature_combination


def GetOptimalFeatureCombination(all_X, global_y, train_index, all_features, feature_combinations):
    
    global_y_train=global_y[train_index]
    all_X_train=[]
    for X_temp in all_X:
        all_X_train.append(X_temp[train_index])
    
    max_auc=0
    max_accuracy=0 
    max_sensitivity=0 
    max_specificity=0
    for combination in feature_combinations:
        auc_score, accuracy, sensitivity, specificity=InnerCrossValidation(all_X_train, global_y_train, all_features, combination)       
        if auc_score>max_auc:
            max_auc=auc_score
            max_accuracy=accuracy
            max_sensitivity=sensitivity
            max_specificity=specificity
            optimal_combination=combination
        elif auc_score==max_auc:
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                max_sensitivity=sensitivity
                max_specificity=specificity
                optimal_combination=combination
            elif accuracy==max_accuracy:
                if sensitivity+specificity>max_sensitivity+max_specificity:
                    max_sensitivity=sensitivity
                    max_specificity=specificity
                    optimal_combination=combination
                    
    return  optimal_combination
        
            
def InnerCrossValidation(X_list, y, all_features, combination):  
    folds = KFold(len(y),n_folds=5,shuffle=True,random_state=np.random.RandomState(1))
    clf=RandomForestClassifier(random_state=1,n_estimators=50)
    predicted_probability=-np.ones(len(y))
    predicted_label=-np.ones(len(y))

    for train_index, test_index in folds:       
        X_train, X_test, y_train=ConstructTrainAndTestDataset(X_list, y, combination, train_index, test_index)   
        probability_test =(clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index]=probability_test[:, 1]
        predicted_label[test_index]=(clf.fit(X_train, y_train)).predict(X_test) 
        
    fpr, tpr, thresholds = roc_curve(y, predicted_probability,pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(y,predicted_label)
    sensitivity=recall_score(y,predicted_label)
    specificity=(accuracy*len(y)-sensitivity*sum(y))/(len(y)-sum(y))
    
    print('Inner cross validation on training dataset, results for feature combination:')
    print(list(array(all_features)[combination]))
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****\n'%(auc_score, accuracy, sensitivity, specificity))
    
    return auc_score, accuracy, sensitivity, specificity           


def ConstructTrainAndTestDataset(X_list, y, combination, train_index, test_index):
    
    y_train=y[train_index]
    
    for i in range(len(combination)):
        if combination[i]==3:
            vdim=30
            train_seqs=X_list[combination[i]][train_index]
            test_seqs=X_list[combination[i]][test_index]           
            pssm=getPssmMatrix(train_seqs, y_train, vdim)
            X_train_temp=getFeatureFromPssm(train_seqs, pssm, vdim)
            X_test_temp=getFeatureFromPssm(test_seqs, pssm, vdim)
        else:
            X_train_temp=X_list[combination[i]][train_index]
            X_test_temp=X_list[combination[i]][test_index]
        if i==0:
            X_train=X_train_temp
            X_test=X_test_temp
        else:
            X_train=np.hstack((X_train, X_train_temp))
            X_test=np.hstack((X_test, X_test_temp))
            
    return X_train, X_test, y_train
              
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


########################################################################################

if __name__ == '__main__':
    
    #input sequences for pssm feature    
    fp=open("posi_samples.fasta",'r')
    posis=getSequences(fp)
    fn=open("nega_samples.fasta",'r')
    negas=getSequences(fn)   
    instances=array(posis+negas)
    global_y=array([1]*len(posis)+[0]*len(negas))   
    print('The number of positive and negative samples: %d,%d'%(len(posis),len(negas))) 
    
    #input all coded feature vectors besides pssm feature
    all_features=['SpectrumProfile','MismatchProfile','SubsequenceProfile','Pssm','Psednc']
    all_X=[]    
    for feature in all_features:
        if feature!='Pssm':
            X=np.loadtxt(feature+'Feature.txt')
            all_X.append(X)
        else:
            all_X.append(instances)

    #all feature combinations   
    feature_combinations=[]
    for i in range(1,len(all_features)+1):
        feature_combinations+=list(itertools.combinations(range(len(all_features)), i)) 
    feature_combinations=[list(feature_combinations[i]) for i in range(len(feature_combinations))]
    
    #prediction based on direct combination model
    tic=time.clock()
    
    print('The prediction based on direct combination model, beginning')
    print('This process may spend several hours, please do not close the program')
    
    classifier=RandomForestClassifier(random_state=1,n_estimators=500)
    folds = KFold(len(global_y),n_folds=10,shuffle=True,random_state=np.random.RandomState(1))
    auc_score, accuracy, sensitivity, specificity, unique_optimal_feature_combination = \
               globalCrossValidation(all_X, global_y, all_features, feature_combinations, classifier, folds)       
    
    print('############################################################################')
    print('results for global cross validation, optimal feature combinations: ',unique_optimal_feature_combination)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****'%(auc_score, accuracy, sensitivity, specificity))
    
    toc=time.clock()    
    print('total running time:%.3f hour'%((toc-tic)/3600))
    print('############################################################################\n')
    
    #output results
    results=DataFrame({'Optimal feature subset':[unique_optimal_feature_combination],\
                      'AUC':[auc_score],\
                      'ACC':[accuracy],\
                      'SN':[sensitivity],\
                      'SP':[specificity]})   
    results=results[['Optimal feature subset','AUC','ACC','SN','SP']]
    results.to_csv('DirectCombinationResults.csv',index=False)
    
