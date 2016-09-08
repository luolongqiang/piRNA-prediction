# -*- coding: utf-8 -*-
#author: longqiang luo

import numpy as np
from numpy import array
import time
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

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

def GetIndividualFeatureBasedModel(all_X, y, all_features, clf):
    for i in range(len(all_features)):
        feature=all_features[i]
        X=all_X[i]
        print("..........................................................................")
        print("Based on "+feature+" model")
        tic=time.clock()
        if feature=="Pssm":
            pssm=GetPssmMatrix(X, y, vdim)
            np.savetxt("PssmMatrix"+animal+".txt",pssm)
            X=GetFeatureFromPssm(X, pssm, vdim)
        classifier=clf.fit(X,y)
        joblib.dump(classifier, feature+animal+".model")
        toc=time.clock()
        print("Based on "+feature+",running time:"+str((toc-tic)/60.0)+" minutes")
        print('..........................................................................\n')

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

    global animal
    global vdim
    vdim=35                        #the fixed length of sequences for the PSSM feature 
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

    clf=RandomForestClassifier(random_state=1,n_estimators=n_trees) 
    GetIndividualFeatureBasedModel(all_X, y, all_features, clf)
