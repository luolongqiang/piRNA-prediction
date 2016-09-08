# -*- coding: utf-8 -*-
#author: longqiang luo

import numpy as np
from numpy import array
from itertools import combinations, combinations_with_replacement, permutations
from repDNA.nac import RevcKmer
from repDNA.psenac import PCPseDNC,PCPseTNC,SCPseDNC,SCPseTNC
import multiprocessing
import time
import sys

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

def GetKmerDict(alphabet,k):
    kmerlst=[]
    partkmers=list(combinations_with_replacement(alphabet, k))
    for element in partkmers:
        elelst=set(permutations(element, k))
        strlst=[''.join(ele) for ele in elelst]
        kmerlst+=strlst
    kmerlst=np.sort(kmerlst)
    kmerdict={kmerlst[i]:i for i in range(len(kmerlst))}
    return kmerdict
  
############################### Spectrum Profile ##############################
def GetSpectrumProfile(instances, alphabet, k):
    kmerdict=GetKmerDict(alphabet, k)
    X=[]
    for sequence in instances:
        vector=GetSpectrumProfileVector(sequence, kmerdict, k)
        X.append(vector)
    X=array(X)    
    return X

def GetSpectrumProfileVector(sequence, kmerdict, k):    
    vector=np.zeros((1,len(kmerdict)))
    n=len(sequence)
    for i in range(n-k+1):
        subsequence=sequence[i:i+k]
        position=kmerdict.get(subsequence)
        vector[0,position]+=1
    return list(vector[0])
    
############################### Mismatch Profile ##############################
def GetMismatchProfile(instances, alphabet, k, m):
    kmerdict=GetKmerDict(alphabet, k)
    X=[]
    for sequence in instances:
        vector=GetMismatchProfileVector(sequence, alphabet, kmerdict, k)
        X.append(vector)  
    X=array(X)    
    return X

def GetMismatchProfileVector(sequence, alphabet, kmerdict, k):    
    vector=np.zeros((1,len(kmerdict)))
    n=len(sequence)
    for i in range(n-k+1):
        subsequence=sequence[i:i+k]
        position=kmerdict.get(subsequence)
        vector[0,position]+=1
        for j in range(k):
            substitution=subsequence
            for letter in set(alphabet)^set(subsequence[j]):
                substitution=list(substitution)
                substitution[j]=letter
                substitution=''.join(substitution)
                position=kmerdict.get(substitution)
                vector[0,position]+=1    
    return list(vector[0])

############################# Subsequence Profile ############################# 
def GetSubsequenceProfileByParallel(instances, alphabet, k, delta):
    cpu_num=multiprocessing.cpu_count()   
    batches=ConstructPartitions(instances, cpu_num)
    pool = multiprocessing.Pool(cpu_num)
    results=[]
    for batch in batches:
        temp=pool.apply_async(GetSubsequenceProfile, (batch, alphabet, k, delta))
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
    
def GetSubsequenceProfile(instances, alphabet, k, delta):
    kmerdict=GetKmerDict(alphabet, k)
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
    
########################### Reverse Compliment Kmer ###########################
def GetRevcKmer(k):
    rev_kmer = RevcKmer(k=k)
    pos_vec = rev_kmer.make_revckmer_vec(open(posi_samples_file))
    neg_vec = rev_kmer.make_revckmer_vec(open(nega_samples_file))
    X = array(pos_vec + neg_vec)
    return X
############ Parallel Correlation Pseudo Dinucleotide Composition #############
def GetPCPseDNC(lamada, phyche_list):
    pc_psednc = PCPseDNC(lamada=lamada, w=0.05)
    pos_vec = pc_psednc.make_pcpsednc_vec(open(posi_samples_file),phyche_index=phyche_list)
    neg_vec = pc_psednc.make_pcpsednc_vec(open(nega_samples_file),phyche_index=phyche_list)
    X = array(pos_vec + neg_vec)    
    return X

############ Parallel Correlation Pseudo Trinucleotide Composition ############
def GetPCPseTNC(lamada):
    pc_psetnc = PCPseTNC(lamada=lamada, w=0.05)
    pos_vec = pc_psetnc.make_pcpsetnc_vec(open(posi_samples_file), all_property=True)
    neg_vec = pc_psetnc.make_pcpsetnc_vec(open(nega_samples_file), all_property=True)
    X = array(pos_vec + neg_vec)
    return X
    
############## Series Correlation Pseudo Dinucleotide Composition #############
def GetSCPseDNC(lamada, phyche_list):
    sc_psednc = SCPseDNC(lamada=lamada, w=0.05)
    pos_vec = sc_psednc.make_scpsednc_vec(open(posi_samples_file), phyche_index=phyche_list)
    neg_vec = sc_psednc.make_scpsednc_vec(open(nega_samples_file), phyche_index=phyche_list)
    X = array(pos_vec + neg_vec)
    return X  
    
############## Series Correlation Pseudo Trinucleotide Composition ############
def GetSCPseTNC(lamada):
    sc_psetnc = SCPseTNC(lamada=lamada, w=0.05)
    pos_vec = sc_psetnc.make_scpsetnc_vec(open(posi_samples_file), all_property=True)
    neg_vec = sc_psetnc.make_scpsetnc_vec(open(nega_samples_file), all_property=True)
    X = array(pos_vec + neg_vec)
    return X 

############################### Sparse Profile ################################
def GetSparseProfile(instances,alphabet,vdim):
    sparse_dict=GetSparseDict(alphabet)
    X=[]
    for sequence in instances:
        vector=GetSparseProfileVector(sequence,sparse_dict,vdim)
        X.append(vector)
    X=array(X)
    return X

def GetSparseDict(alphabet):
    alphabet_num=len(alphabet)
    identity_matrix=np.eye(alphabet_num+1)
    sparse_dict={alphabet[i]:identity_matrix[i] for i in range(alphabet_num)}
    sparse_dict['E']=identity_matrix[alphabet_num]
    return sparse_dict

def GetSparseProfileVector(sequence,sparse_dict,vdim):
    seq_length=len(sequence)
    sequence=sequence+'E'*(vdim-seq_length) if seq_length<=vdim else sequence[0:vdim]   
    vector=sparse_dict.get(sequence[0])
    for i in range(1,vdim):
        temp=sparse_dict.get(sequence[i])
        vector=np.hstack((vector,temp))
    return vector

######################## Position-Specific Scoring Matrix #####################
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
    
###############################################################################
    
if __name__ == '__main__':

    global posi_samples_file
    global nega_samples_file
    posi_samples_file=sys.argv[1]
    nega_samples_file=sys.argv[2]
    animal=posi_samples_file.split('_')[0]
    fp=open(posi_samples_file,'r')
    posis=GetSequences(fp)
    fn=open(nega_samples_file,'r')
    negas=GetSequences(fn)   
    instances=array(posis+negas)
    alphabet=['A','C','G','T']
    
    # Spectrum Profile for k=1,2,3,4,5
    for k in range(1,6):
        print('..........................................................................')
        print('Coding for feature:'+str(k)+'-Spectrum Profile, beginning')
        tic=time.clock()
        X=GetSpectrumProfile(instances,alphabet,k)
        np.savetxt(str(k)+'-SpectrumProfileFeature'+animal+'.txt',X)
        toc=time.clock()
        print('Coding time:%.3f minutes'%((toc-tic)/60))
        
    # Mismatch Profile for (k,m)=(3,1),(4,1),(5,1)
    for (k,m) in [(3,1),(4,1),(5,1)]:
        print('..........................................................................')
        print('Coding for feature:'+str((k,m))+'-Mismatch Profile, beginning')
        tic=time.clock()
        X=GetMismatchProfile(instances,alphabet,k,m)
        np.savetxt(str((k,m))+'-MismatchProfileFeature'+animal+'.txt',X)
        toc=time.clock()
        print('Coding time:%.3f minutes'%((toc-tic)/60))    

    # Subsequence Profile for (k,delta)=(3,1),(4,1),(5,1)
    for (k,delta) in [(3,1),(4,1),(5,1)]:
        print('..........................................................................')
        print('Coding for feature:'+str((k,delta))+'-Subsequence Profile, beginning')
        print('The process may spend some time, please do not close the program')
        tic=time.clock()
        X=GetSubsequenceProfileByParallel(instances,alphabet,k,delta)
        np.savetxt(str((k,delta))+'-SubsequenceProfileFeature'+animal+'.txt',X)
        toc=time.clock()
        print('Coding time:%.3f minutes'%((toc-tic)/60)) 
        
    # Reverse Compliment Kmer for k=1,2,3,4,5
    for k in range(1,6):
        print('..........................................................................')
        print('Coding for feature:'+str(k)+'-RevcKmer, beginning')
        tic=time.clock()
        X=GetRevcKmer(k)
        np.savetxt(str(k)+'-RevcKmerFeature'+animal+'.txt',X)
        toc=time.clock()
        print('Coding time:%.3f minutes'%((toc-tic)/60))
        
    # Parallel Correlation Pseudo Dinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:PCPseDNC, beginning')
    tic=time.clock()
    X=GetPCPseDNC(1,phyche_list=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    np.savetxt('PCPseDNCFeature'+animal+'.txt',X)
    toc=time.clock()
    print('Coding time:%.3f minutes'%((toc-tic)/60))

    # Parallel Correlation Pseudo Trinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:PCPseTNC, beginning')
    tic=time.clock()
    X=GetPCPseTNC(1)
    np.savetxt('PCPseTNCFeature'+animal+'.txt',X)
    toc=time.clock()
    print('Coding time:%.3f minutes'%((toc-tic)/60))
    
    # Series Correlation Pseudo Dinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:SCPseDNC, beginning')
    tic=time.clock()
    X=GetSCPseDNC(1,phyche_list=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    np.savetxt('SCPseDNCFeature'+animal+'.txt',X)
    toc=time.clock()
    print('Coding time:%.3f minutes'%((toc-tic)/60))
    
    # Series Correlation Pseudo Trinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:SCPseTNC, beginning')
    tic=time.clock()
    X=GetSCPseTNC(1)
    np.savetxt('SCPseTNCFeature'+animal+'.txt',X)
    toc=time.clock()
    print('Coding time:%.3f minutes'%((toc-tic)/60))  
    
    # Sparse Profile
    print('..........................................................................')
    print('Coding for feature:Sparse Profile, beginning')
    tic=time.clock()
    X=GetSparseProfile(instances,alphabet,vdim=35)
    np.savetxt('SparseProfileFeature'+animal+'.txt',X)
    toc=time.clock()
    print('Coding time:%.3f minutes'%((toc-tic)/60))
    
    # Position-Specific Scoring Matrix(PSSM)
    # the PSSM feature is dependent on the positive sequences on training set
    
    # all features:
    #'1-SpectrumProfile','2-SpectrumProfile','3-SpectrumProfile','4-SpectrumProfile','5-SpectrumProfile',\
    #'(3, 1)-MismatchProfile','(4, 1)-MismatchProfile','(5, 1)-MismatchProfile',\
    #'(3, 1)-SubsequenceProfile','(4, 1)-SubsequenceProfile','(5, 1)-SubsequenceProfile',\
    #'1-RevcKmer','2-RevcKmer','3-RevcKmer','4-RevcKmer','5-RevcKmer',\
    #'PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC','SparseProfile','Pssm'
    