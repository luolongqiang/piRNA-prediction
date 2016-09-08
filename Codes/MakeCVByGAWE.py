# -*- coding: utf-8 -*-
#author: longqiang luo

import numpy as np
from numpy import array
from pandas import DataFrame
import random
import time
import sys

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

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
        
def ConductCrossValidation(all_X, y, all_features, clf, folds):
    predicted_proba = -np.ones(len(y))
    predicted_label = -np.ones(len(y))
    all_optimal_weights=[]
    cv_round=1
    for train_index, vali_index, test_index in folds:
        print('..........................................................................')
        print('Cross validation, round %d, beginning'%cv_round)
        start=time.clock()
        
        vali_proba_matrix, y_vali, test_proba_matrix, test_label_matrix= \
        GetIndividualFeatureResults(all_X, y, all_features, clf, train_index, vali_index, test_index)
       
        optimal_weights=GeneticAlgorithm(vali_proba_matrix, y_vali)
        all_optimal_weights.append(optimal_weights)
        
        combined_proba=np.dot(test_proba_matrix, optimal_weights)
        combined_label=np.dot(test_label_matrix, optimal_weights)>0.5
        
        predicted_proba[test_index]=combined_proba
        predicted_label[test_index]=combined_label        

        end=time.clock()       
        print('The optimal weights:\n',optimal_weights)
        print('Round %d, running time: %.3f hour'%(cv_round, (end-start)/3600))
        print('..........................................................................\n')            
        cv_round+=1
        
    auc_score, accuracy, sensitivity, specificity=EvaluatePerformances(y,predicted_proba,predicted_label)
    all_optimal_weights=array(all_optimal_weights)    
    return auc_score, accuracy, sensitivity, specificity, all_optimal_weights

def GetIndividualFeatureResults(all_X, y, all_features, clf, train_index, vali_index, test_index):
    vali_proba_matrix=[]
    test_proba_matrix=[]
    test_label_matrix=[]
    for i in range(len(all_features)):
        feature=all_features[i]
        X=all_X[i]
        
        X_train,X_vali,X_test,y_train,y_vali=GetPartitionOfSamples(X,y,feature,train_index,vali_index,test_index)
        
        classifier=clf.fit(X_train, y_train)
        temp_vali_proba=classifier.predict_proba(X_vali)
        temp_test_proba=classifier.predict_proba(X_test)
        temp_test_label=classifier.predict(X_test)
        
        vali_proba_matrix.append(temp_vali_proba[:,1])        
        test_proba_matrix.append(temp_test_proba[:,1])      
        test_label_matrix.append(temp_test_label)
        
    vali_proba_matrix=np.transpose(vali_proba_matrix)
    test_proba_matrix=np.transpose(test_proba_matrix)
    test_label_matrix=np.transpose(test_label_matrix)  
    return vali_proba_matrix, y_vali, test_proba_matrix, test_label_matrix
        
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
 
def GeneticAlgorithm(vali_proba_matrix, y_vali):
    global pops_num
    global generations
    global chr_length
    pops=GetPopulations(pops_num, chr_length)
    auc_scores=FitnessFunction(pops, vali_proba_matrix, y_vali)    
    for k in range(generations):
        pops=Updatepops(pops,auc_scores)
        auc_scores=FitnessFunction(pops, vali_proba_matrix, y_vali)
    max_auc=np.max(auc_scores)
    print('The maximum AUC is %.3f on validation dataset'%max_auc)
    max_index=list(auc_scores).index(np.max(auc_scores))
    optimal_weights=pops[max_index]    
    return optimal_weights

def GetPopulations(pops_num, chr_length):
    pops=[]
    for i in range(pops_num-chr_length):
        temp_pop=[random.uniform(0,1) for i in range(chr_length)]
        temp_pop=temp_pop/np.sum(temp_pop)
        pops.append(temp_pop)
    pops=array(pops)
    pops=np.vstack((np.eye(chr_length), pops))
    return pops    
    
def FitnessFunction(pops, vali_proba_matrix, y_vali):
    auc_scores=[]
    for i in range(np.shape(pops)[0]):
        weights=pops[i]
        combined_mean_proba=np.dot(vali_proba_matrix, weights)
        fpr,tpr,thresholds = roc_curve(y_vali, combined_mean_proba, pos_label=1)
        auc_scores.append(auc(fpr, tpr))          
    auc_scores=array(auc_scores)
    return auc_scores    

def Updatepops(pops,auc_scores):
    global pops_num
    new_order=random.sample(range(pops_num),pops_num)
    for i in np.linspace(0,pops_num,num=pops_num/2,endpoint=False,dtype=int):
        fmax=np.max(auc_scores)
        fmin=np.min(auc_scores)
        fmean=np.mean(auc_scores)
        
        select_index=new_order[i:i+2]        
        f=np.max(auc_scores[select_index])  
        two_pops=pops[select_index].copy()
        
        probacrossover=(fmax-f)/(fmax-fmean) if f>fmean else 1
        cross_pops=Crossover(two_pops) if probacrossover>random.uniform(0,1) else two_pops.copy()          

        probamutation=0.5*(fmax-f)/(fmax-fmean) if f>fmean else (fmean-f)/(fmean-fmin)
        new_two_pops=Mutation(cross_pops) if probamutation>random.uniform(0,1) else cross_pops.copy()                    
      
        pops[select_index]=new_two_pops.copy()  
    return pops

def Crossover(two_pops):
    global chr_length
    cross_pops=two_pops.copy()
    crossposition=random.randint(2,chr_length-3)    
    cross_pops[0][0:crossposition]=two_pops[1][0:crossposition]
    cross_pops[1][0:crossposition]=two_pops[0][0:crossposition] 
    cross_pops=Normalize(cross_pops)
    return cross_pops

def Mutation(cross_pops):
    global chr_length
    new_two_pops=cross_pops.copy()
    for i in range(2):
        mutation_num=random.randint(1,round(chr_length/5))
        mutation_positions=random.sample(range(chr_length),mutation_num)
        new_two_pops[i][mutation_positions]=[random.uniform(0,1) for j in range(mutation_num)]
    new_two_pops=Normalize(new_two_pops)     
    return new_two_pops 

def Normalize(two_pops):
    global chr_length
    for i in range(2):     
        if np.sum(two_pops[i])<10**(-12):
            two_pops[i]=[random.uniform(0,1) for j in range(chr_length)]
        two_pops[i]=two_pops[i]/np.sum(two_pops[i])
    return two_pops
    
def EvaluatePerformances(real_label,predicted_proba,predicted_label):
    fpr, tpr, thresholds = roc_curve(real_label, predicted_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy=accuracy_score(real_label,predicted_label)
    sensitivity=recall_score(real_label,predicted_label)
    specificity=(accuracy*len(real_label)-sensitivity*sum(real_label))/(len(real_label)-sum(real_label))    
    return auc_score, accuracy, sensitivity, specificity 
    
########################################################################################

if __name__ == '__main__':
    all_features=['1-SpectrumProfile','2-SpectrumProfile','3-SpectrumProfile','4-SpectrumProfile','5-SpectrumProfile',\
                  '(3, 1)-MismatchProfile','(4, 1)-MismatchProfile','(5, 1)-MismatchProfile',\
                  '(3, 1)-SubsequenceProfile','(4, 1)-SubsequenceProfile','(5, 1)-SubsequenceProfile',\
                  '1-RevcKmer','2-RevcKmer','3-RevcKmer','4-RevcKmer','5-RevcKmer',\
                  'PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC','SparseProfile','Pssm']
          
    global chr_length
    global pops_num
    global generations
    global vdim
    chr_length=len(all_features)   #the length of chromosomes for the genetic algorithm
    pops_num=1000                  #the population size for the genetic algorithm 
    generations=500                #the generation for the genetic algorithm
    vdim=35                        #the fixed length of sequences for the PSSM feature 
    folds_num=10                   #the number of folds for the cross validation
    seeds_num=1                   #the number of seeds for the partition of dataset
    n_trees=500                    #the number of trees for the random forest
    
    posi_samples_file=sys.argv[1]
    nega_samples_file=sys.argv[2]
    animal=posi_samples_file.split('_')[0]
    fp=open(posi_samples_file,'r')
    posis=GetSequences(fp)
    fn=open(nega_samples_file,'r')
    negas=GetSequences(fn)   
    instances=array(posis+negas)
    y=array([1]*len(posis)+[0]*len(negas))    
    print('The number of positive and negative samples: %d,%d'%(len(posis),len(negas)))    
    
    all_X=[]    
    for feature in all_features:
        if feature!='Pssm':
            X=np.loadtxt(feature+'Feature'+animal+'.txt')
            all_X.append(X)
        else:
            all_X.append(instances)     
    
    results=[] 
    clf=RandomForestClassifier(random_state=1,n_estimators=n_trees)         
    for seed in range(1,seeds_num+1):
        print('################################## Seed %d ##################################'%seed)
        print('The prediction using GA-based ensemble learning, beginning')
        print('This process may spend some time, please do not close the program')
        tic=time.clock()
              
        folds=ConstructPartitionOfSet(y, folds_num, seed)    
        auc_score, accuracy, sensitivity, specificity, all_optimal_weights=\
                   ConductCrossValidation(all_X, y, all_features, clf, folds)  
        results.append([auc_score, accuracy, sensitivity, specificity])
        
        toc=time.clock()
        print('**************************************************************************')
        print('Seed %d, the final predicted results:'%seed)
        print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f****'\
                           %(auc_score, accuracy, sensitivity, specificity))
        print('Total running time:%.3f hour'%((toc-tic)/3600))
        print('**************************************************************************\n')    
        
        feature_weights=DataFrame({'Round '+str(i+1):all_optimal_weights[i,:] for i in range(folds_num)})
        feature_weights['Feature']=all_features
        feature_weights=feature_weights[['Feature']+['Round '+str(i+1) for i in range(folds_num)]]        
        feature_weights.to_csv('OptimalFeatureWeights(seed%d).csv'%seed,index=False)
    
    results=array(results)    
    df=DataFrame({'Seed':range(1,seeds_num+1),\
                  'AUC':results[:,0],\
                  'ACC':results[:,1],\
                  'SN':results[:,2],\
                  'SP':results[:,3]}) 
    df=df[['Seed','AUC','ACC','SN','SP']]
    df.to_csv('Results'+animal+'CV(GAWE).csv',index=False)
