# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 19:48:31 2017

@author: Administrator
"""

import srForestTrain
import sys
import numpy as np
import math
import warnings
#import random

#get additional parameters and fill in remaining parameters
def forestRegrTrain( Xfeat, Xsrc, Xtar, varargin ):
    dfs={ 'M':1,  'minChild':64,  'minCount':128,  'N1':[],  'F1':[],  'F2':5,
         'maxDepth':64,  'fWts':[],  'splitfuntype':'pair', 
      'nodesubsample':1000,  'splitevaltype':'variance', 
      'lambda':0.01,  'estimatelambda':0,  'kappa':1,  'leaflearntype':'linear',
      'usepf':0,  'verbose':0 }
    opts=srForestTrain.getPrmDflt(varargin,dfs,1)
    if len(sys.argv)==0:
        forest=opts
        return forest
    
    Ff,N=np.shape(Xfeat)
    Ncheck=Xsrc.shape[1]
    assert(N==Ncheck)
    Ncheck=Xtar.shape[1]
    assert(N==Ncheck)
    if len(opts['N1'])==0:
        opts['N1']=round(N*0.75)
        opts['N1']=min(N,opts['N1'])
    if len(opts['F1'])==0:
        opts['F1']=round(math.sqrt(Ff))
        opts['F1']=min(Ff,opts['F1'])
    if opts['F2']<0:
        raise np.error('F2 should be > -1')
    if opts['fWts']==[]:
        opts['fWts']=np.ones((1,Ff),np.float32)
        opts['fWts']=opts['fWts']/sum(sum(opts['fWts']))
    if opts['nodesubsample']<opts['minChild']*2:
        raise np.error('nodesubsample < 2*minChild')
    if opts['nodesubsample']<opts['minChild']*3:
        warnings.warn('nodesubsample < 3*minChild',DeprecationWarning)

#make sure data has correct types
    pass
    
#train M random trees on different subsets of data
    dWtsUni=np.ones((1,N),np.float32)
    dWtsUni=dWtsUni/sum(sum(dWtsUni))
    if opts['usepf']==1:
        tem_forest=[]
        for i in range(opts['M']):
           if N==opts['N1']:
               d=np.arange(0,N)
           else:
               d=wswor(dWtsUni,opts.N1,4)
           Xfeat1=Xfeat[:,d]
           Xsrc1=Xsrc[:,d]
           Xtar1=Xtar[:,d]
           temforest=treeRegrTrain(Xfeat1,Xsrc1,Xtar1,opts)
           tem_forest.append(temforest)
        forest=[]
        #forest=tem_forest
        for i in range(opts['M']):
            forest.append(tem_forest[i])
    else:
        for i in range(opts['M']):
            if N==opts['N1']:
                d=np.arange(0,N)
            else:
                d=wswor(dWtsUni,opts.N1,4)
            Xfeat1=Xfeat[:,d]
            Xsrc1=Xsrc[:,d]
            Xtar1=Xtar[:,d]
            tree=treeRegrTrain(Xfeat1,Xsrc1,Xtar1,opts)
            forest=[]
            forest=tree
    
    return forest
                


# Train a single regression tree
def treeRegrTrain( Xfeat, Xsrc, Xtar, opts ):
    
 #    define some constants and the tree model
    N=Xfeat.shape[1]
    K=2*N-1
    thrs=np.zeros((K,1),np.float32)
    if opts['splitfuntype']=='single':
        fids=np.zeros((K,1),np.int32)
    elif opts['splitfuntype']=='pair':
        fids=np.zeros((K,2),np.int32)
    else:
        raise np.error('Unknown splitfunction type')
    child=np.zeros((K,1),np.int32)
    count=child
    depth=child
    leaf_info={'T':[],'type':-1,'id':-1}
#     leafinfo=leafinfo(np.ones((K,1)))
    "map函数的使用？"
    leafinfo=[]
    for i in range(K):
        leafinfo.append(leaf_info)
    dids=[]
    for j in range(K):
        dids.append([])
    dids[0]=np.arange(K)
    k=0
    K=2
    msgnodestep=200
# train the tree   
    while k<K:
        dids1=dids[k]
        count[k]=len(dids1)
        XfeatNode=Xtar[:,dids1]
        XsrcNode=Xsrc[:,dids1]
        XtarNode=Xtar[:,dids1]
        if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
            print('Node %d, depth %d, %d samples () ',k,depth[k],count[k])
        if count[k]<=opts['minCount'] or depth[k]>opts['maxDepth'] or count[k]<(2*opts['minChild']):
            if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
                print('becomes a leaf (stop criterion active)\n')
            leafinfo[k]=createLeaf(XsrcNode,XtarNode,opts['leaflearntype'],opts['lambda'],opts['estimatelambda'])
            k=k+1
    #     continue
        if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
            print('find split () ')
             
    #compute responses for all data samples
        if opts['splitfuntype']=='single':
            pass
        elif opts['splitfuntype']=='pair':
            fids1 = np.array([wswor(opts['fWts'],opts['F1'],4), wswor(opts['fWts'],opts['F1'],4)])
    # Caution: same feature id could be sampled  -> all zero responses
            resp=XfeatNode[fids1[0,:],:]-XfeatNode[fids1[1,:],:]
        else:
            raise np.error('Unknown splitfunction type')
    #subsample the data for splitfunction node optimization 随机选样本
        if opts['nodesubsample']>0 and opts['nodesubsample']<count[k]:
            randinds=np.random.permutation(count[k])[0:opts['nodesubsample']]
            respSub=resp[:,randinds]
            XsrcSub=XsrcNode[:,randinds]
            XtarSub=XtarNode[:,randinds]
        else:
            respSub = resp
            XsrcSub = XsrcNode
            XtarSub = XtarNode
        if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
            print('subsmpl = %07d/%07d () ',respSub.shape[1],resp.shape[1])
    #find best splitting function and corresponding threshold寻找最优的节点分裂函数和相对应的阈值
        [fid,thr,rerr]=findSplitAndThresh(respSub,XsrcSub,XtarSub,opts['F2'],opts['splitevaltype'],opts['lambda'],opts['minChild'],opts['kappa'])
    #check validity of the splitting function
        validsplit=0
        left=resp[fid,:]<thr
        count0=np.count_nonzero(left)
        fid=fids1[:,fid]
        if rerr!=np.inf and count0>=opts['minChild'] and (count[k]-count0)>=opts['minChild']:
            validsplit=1
        if validsplit==1:
            child[k]=K
            fids[k,:]=fid
            thrs[k]=thr
            dids[K-1]=dids1[left]
            dids[K]=dids[~left]
            depth[K-1:K+1]=depth[K]+1
            K=K+2
            dids[k]=[]
            if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
                print('valid split (loss=%.6f)\n',rerr)
        else:
            if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
                print('invalid split -> leaf\n')
            leafinfo[k]=createLeaf(XsrcNode,XtarNode,opts['leaflearntype'],opts['lambda'],opts['estimatelambda'])
        k=k+1
#    K=K-1
#create output model struct
    tree={'fids':fids[0:K,:],'thrs':thrs[0:K],'child':child[0:K],'count':count[0:K],'depth':depth[0:K],'leafinfo':leafinfo[0:K],'dids':[]}
#create the leaf-node id mapping
    leafcnt = 0
    for i in range(len(tree['leafinfo'])):
        if len(tree['leafinfo'][i]['T'])!=0:
            leafcnt=leafcnt+1
            tree['leafinfo'][i]['id']=leafcnt
    
    return tree
    
def wswor( prob, N, trials ):
#Fast weighted sample without replacement. Alternative to:
    M=prob.shpae[1]
    assert(N<=M)
    if N==M:
        pass
    for i in range(M):
        assert(prob[i]==prob[0])
    ids=np.random.permutation(30)
    ids=ids[0:N]
    return ids


def findSplitAndThresh( resp, Xsrc, Xtar, F2, splitevaltype, lamdda, minChild, kappa ):
    F1=resp.shape[0]
    rerr=float("inf")
    fid=1
    thr=float("inf")
    Ft=Xtar.shape[0]
    Fs=Xsrc.shape[0]
#special treatment for random tree growing
    if splitevaltype=='random':
        F1=1
        F2=1
    for s in range(F1):
#get thresholds to evaluate
        if F2==0:
            tthrs=np.median(resp[s,:],axis=0)
        else:
            respmin=min(resp[s,:])
            respmax=max(resp[s,:])
            tthrs=np.zeros((F2+1,1),np.float32)
            tthrs[0:-1]=np.random.rand(1,5)*0.95*(respmax-respmin)+respmin
            tthrs[-1]=np.median(resp[s,:])
        for t in range(len(tthrs)):
            tthr=tthrs[t]
            left=resp[s,:]<tthr
            right=~left
            nl=0
            nr=0
            for i in range(len(resp)):
                if left[i]==1:
                    nl=nl+1
                else:
                    nr=nr+1
#            left=left.astype('int')
#            right=right.astype('int')
            if nl<minChild or nr<minChild:
                continue
#  mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:] #nonzero对应于去掉特征数据缺失值
#  mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
            XsrcL=Xsrc[:,left]
            XsrcR=Xsrc[:,right]
            XtarL=Xtar[:,left]
            XtarR=Xtar[:,right]
            if splitevaltype=='random':
                trerr=0
            elif splitevaltype=='banlanced':
                trerr=np.square(nl-nr)
            elif splitevaltype=='variance':
                trerrL = sum(np.var(XtarL,axis=1,ddof=1))/Ft
                trerrR = sum(np.var(XtarR,axis=1,ddof=1))/Ft
                if kappa>0:
                    trerrLsrc=sum(np.var(XsrcL,axis=1,ddof=1))/Fs
                    trerrRsrc=sum(np.var(XsrcR,axis=1,ddof=1))/Fs
                    trerrL=(trerrL+kappa*trerrLsrc)/2
                    trerrR=(trerrR+kappa*trerrRsrc)/2
                trerr = (nl*trerrL + nr*trerrR)/(nl+nr)
            elif splitevaltype=='reconstruction':
                XsrcL=np.row_stack((XsrcL,np.ones(XsrcL.shape[1])))
                TL=XtarL.dot(np.linalg.lstsq(((XsrcL.dot(XsrcL.T))+lamdda*np.eye(XsrcL.shape[0])),XsrcL)[0].T)
                XsrcR=np.row_stack((XsrcR,np.ones(XsrcR.shape[1])))
                TR=XtarR.dot(np.linalg.lstsq(((XsrcR.dot(XsrcR.T))+lamdda*np.eye(XsrcR.shape[0])),XsrcR)[0].T)
                trerrL = np.sqrt(sum(sum((XtarL-TL*XsrcL)**2))/nl)
                trerrR = np.sqrt(sum(sum((XtarR-TR*XsrcR)**2))/nr)
                if kappa>0:
                    trerrLsrc=sum(np.var(XsrcL,axis=1,ddof=1))/Fs
                    trerrRsrc=sum(np.var(XsrcR,axis=1,ddof=1))/Fs
                    trerrL=(trerrL+kappa*trerrLsrc)/2
                    trerrR=(trerrR+kappa*trerrRsrc)/2
                trerr = (nl*trerrL + nr*trerrR)/(nl+nr)
            else:
                raise np.error('Unknown split evaluation type')
            if trerr<rerr:
                rerr=trerr
                thr=tthr
                fid=s
    return fid,thr,rerr


#creates a leaf node and computes the prediction model
def createLeaf( Xsrc, Xtar, leaflearntype, lamdda, autolambda ):
    if leaflearntype=='constant':
        I=Xtar.sum(axis=1)/Xtar.shape[1]
        predmodeltype = 0
    elif leaflearntype=='linear':
        Xsrc=np.row_stack((Xsrc,np.ones(Xsrc.shape[1])))
        matinv=Xsrc.dot(Xsrc.T)
        if autolambda==1:
            lamdda=estimateLambda(matinv)
        I=Xtar.dot(np.linalg.lstsq((matinv+lamdda*np.eye(Xsrc.shape[0])),Xsrc)[0].T)
        predmodeltype = 1
    elif leaflearntype=='polynomial':
        Xsrc=np.vstack((Xsrc,np.ones(Xsrc.shape[1],Xsrc**2)))
        matinv=Xsrc.dot(Xsrc.T)
        if autolambda==1:
            lamdda=estimateLambda(matinv)
        I=Xtar.dot(np.linalg.lstsq((matinv+lamdda*np.eye(Xsrc.shape[0])),Xsrc)[0].T)
        predmodeltype = 2
    else:
        raise np.error('Unknown leaf node prediction type')
    leaf={'T':[],'type':-1,'id':-1}
    leaf['T']=I
    leaf['type']=predmodeltype
    leaf['id']=-1
    
    return leaf
        
      
def estimateLambda(matinv):
    pass