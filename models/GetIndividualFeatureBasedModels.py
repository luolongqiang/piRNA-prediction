#**********************
# -*- coding: utf-8 -*-
#author: longqiang luo
#data: 2015-12-20
#**********************

import os, sys, time, argparse, pickle, platform
import numpy as np
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from multiprocessing import Pool, cpu_count      

def GetFileList(FindPath, FlagStr=[]):     
    FileList = []  
    FileNames = os.listdir(FindPath)  
    if len(FileNames) > 0:  
       for fn in FileNames:  
           if len(FlagStr) > 0:  
               if IsSubString(FlagStr,fn):  
                   fullfilename = os.path.join(FindPath,fn)  
                   FileList.append(fullfilename)  
           else:  
               fullfilename=os.path.join(FindPath,fn)  
               FileList.append(fullfilename)  
    if len(FileList)>0:  
        FileList.sort()    
    return FileList 
    
def IsSubString(SubStrList,Str):  
    flag=True  
    for substr in SubStrList:  
        if not (substr in Str):  
            flag=False  
    return flag

def GetModels(args):
    feature_txt, y, output_dir = args[0], args[1], args[2]
    X = np.loadtxt(feature_txt)
    if 'Windows' in platform.system():
        feature_name = feature_txt.split('\\')[-1].split('.')[0]
    else:
        feature_name = feature_txt.split('/')[-1].split('.')[0]
    model_name = os.path.join(output_dir, feature_name.replace('Feature', 'Model') + '.pkl')
    print(".................................................................")
    print("get "+feature_name+"-based model")
    tic=time.clock()
    clf=RandomForestClassifier(random_state=1, n_estimators=500) 
    classifier=clf.fit(X,y)
    pickle.dumps(clf) # , open(model_name, 'wb')
    toc=time.clock()
    print("running time:"+str((toc-tic)/60.0)+" minutes")
    print('..................................................................\n')

########################################################################################
def GetArgs():
    parser = argparse.ArgumentParser(description='get individual feature-based model')
    parser.add_argument('-input', dest='input', 
        help='input derectory of features', default=None, type=str)
    parser.add_argument('-output', dest='output',
        help='output directory of models', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

########################################################################################

if __name__ == '__main__':
    
    args = GetArgs()
    input_dir=args.input
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_features = GetFileList(input_dir, ['Feature'])
    labels_txt = GetFileList(input_dir, ['Label'])[0]
    y = np.loadtxt(labels_txt)

    num = len(all_features)
    pool = Pool(int(cpu_count()*3/4))
    pool.map(GetModels, zip(all_features, [y]*num, [output_dir]*num))
