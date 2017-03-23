# -*- coding: utf-8 -*-
#author: longqiang luo

import numpy as np
from numpy import array
import random
import time
import sys

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

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

def GetOptimalWeights(all_X, y, all_features, clf, folds):
    proba_matrix=[]
    #label_matrix=[]
    for i in range(len(all_features)):
        feature=all_features[i]
        X=all_X[i]
        print("..........................................................................")
        print("Cross validation,based on "+feature+",beginning")
        tic=time.clock()
        proba_vector=-np.ones(len(y))
        #label_vector=-np.ones(len(y))
        k=1
        for train_index, test_index in folds:
            print(".....fold %d....."%k)
            k+=1
            X_train,X_test,y_train=GetPartitionOfSamples(X,y,feature,train_index,test_index)
            classifier=clf.fit(X_train, y_train)
            temp_test_proba=classifier.predict_proba(X_test)
            #temp_test_label=classifier.predict(X_test)
            proba_vector[test_index]=temp_test_proba[:,1]
            #label_vector[test_index]=temp_test_label
        proba_matrix.append(proba_vector)
        #label_matrix.append(label_vector)
        toc=time.clock()
        print("Cross validation,based on "+feature+",running time:"+str((toc-tic)/60.0)+" minutes")
        print('..........................................................................\n')
    proba_matrix=np.transpose(proba_matrix)
    optimal_weights=GeneticAlgorithm(proba_matrix, y)
    np.savetxt("OptimalWeights"+animal+".txt",optimal_weights)
    
        
def GetPartitionOfSamples(X, y, feature, train_index, test_index):
    y_train=y[train_index]
    if feature=='Pssm':        
        train_seqs=X[train_index]
        test_seqs =X[test_index]
        global vdim
        pssm=GetPssmMatrix(train_seqs, y_train, vdim)
        X_train=GetFeatureFromPssm(train_seqs, pssm, vdim)
        X_test =GetFeatureFromPssm(test_seqs, pssm, vdim)
    else:
        X_train=X[train_index]
        X_test =X[test_index]
    return X_train, X_test, y_train

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
 
def GeneticAlgorithm(proba_matrix, y):
    global pops_num
    global generations
    global chr_length
    pops=GetPopulations(pops_num, chr_length)
    auc_scores=FitnessFunction(pops, proba_matrix, y)    
    for k in range(generations):
        pops=UpdatePops(pops,auc_scores)
        auc_scores=FitnessFunction(pops, proba_matrix, y)
    max_auc=np.max(auc_scores)
    print('The maximum AUC is %.3f by using genetic algorithm'%max_auc)
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
    
def FitnessFunction(pops, proba_matrix, y):
    auc_scores=[]
    for i in range(np.shape(pops)[0]):
        weights=pops[i]
        combined_mean_proba=np.dot(proba_matrix, weights)
        fpr,tpr,thresholds = roc_curve(y, combined_mean_proba, pos_label=1)
        auc_scores.append(auc(fpr, tpr))          
    auc_scores=array(auc_scores)
    return auc_scores    

def UpdatePops(pops,auc_scores):
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
    vdim=30                        #the fixed length of sequences for the PSSM feature 
    folds_num=10                   #the number of folds for the cross validation
    seeds_num=1                   #the number of seeds for the partition of dataset
    n_trees=500                    #the number of trees for the random forest
    
    global posi_samples_file
    global nega_samples_file
    global animal
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

    clf=RandomForestClassifier(random_state=1,n_estimators=n_trees) 
        
    for seed in range(1,seeds_num+1):              
        folds=list(KFold(len(y),n_folds=folds_num,shuffle=True,random_state=np.random.RandomState(seed)))    
        GetOptimalWeights(all_X, y, all_features, clf, folds)  
