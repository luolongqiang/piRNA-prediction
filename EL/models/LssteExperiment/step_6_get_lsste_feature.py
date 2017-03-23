# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 22:23:46 2015

@author: luo
"""

def get_feature_dict(fi):
    posi_dict={}
    nega_dict={}
    while True:
        s=fi.readline()
        if not s:
            break
        else:
            temp=s.split('\t')
            sample_type=temp[0]
            vector=temp[1]
            for i in range(2,len(temp)-1):
                vector+='\t'+temp[i]
            if 'posi' in sample_type:
                posi_dict[sample_type]=vector
            elif 'nega' in sample_type:
                nega_dict[sample_type]=vector
    return posi_dict, nega_dict


def get_lsste_feature(sample_type, feature_dict, fo):
    n=len(feature_dict)
    for i in range(1,n+1):
        mark='>'+sample_type+str(i)
        fo.write(mark+'\n')
        fo.write(feature_dict[mark]+'\n')

if __name__=='__main__':
   
    fi=open('step_5_out','r')
    posi_dict, nega_dict = get_feature_dict(fi)
    fi.close()

    fo1=open('posi_lsste_feature.fasta','w')
    get_lsste_feature('posi',posi_dict,fo1)
    fo1.close()

    fo2=open('nega_lsste_feature.fasta','w')
    get_lsste_feature('nega',nega_dict,fo2)
    fo2.close()


