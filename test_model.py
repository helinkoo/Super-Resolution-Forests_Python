# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:14:43 2018

@author: HeLin
"""
import numpy as np
import train_model
import os
from scipy import misc
from skimage import measure
import cv2
from skimage import color

def srForestApply( dataLow, dataHigh, srforest, varargin ):
    dfs={'rmborder':3,'Mhat':[],'nthreads':1}
    opts=train_model.getPrmDflt(dfs,1,varargin)

#check input
    if len(dataLow)==0 and len(dataHigh)==0:
        raise ValueError('Either dataLow or dataHigh has to be provided')
    if len(srforest)==0:
        raise ValueError('the model srForest has to be provided')
    if len(opts['Mhat'])==0:
        opts['Mhat']=len(srforest['model'])

# check if we need to downscale the high-res images first for evaluation!
    downscale=0
    if len(dataLow)==0:
        downscale=1
        dataLow=dataHigh
    if isinstance(dataLow,str):
        imlistLow=os.listdir(dataLow)
        nimgs=len(imlistLow)
    else:
        nimgs=1
    if isinstance(dataHigh,str):
        imlistHigh=os.listdir(dataHigh)

    out=[]
    for i in range(nimgs):
        if srforest['sropts']['verbose']==1:
            print('Upscale image %d/%d\n'%(i+1,nimgs))
        if isinstance(dataLow,str):
            imL=misc.imread(os.path.join(dataLow,imlistLow[i]))
        else:
            imL=dataLow
        imLY,imLCB,imLCR=train_model.imageTransformColorT(imL)
        imLY=train_model.imageModcrop(imLY,srforest['sropts']['sf'])
        if len(imLCB)!=0 and len(imLCR)!=0:
            imLCB=train_model.imageModcrop(imLCB,srforest['sropts']['sf'])
            imLCR=train_model.imageModcrop(imLCR,srforest['sropts']['sf'])
        if downscale==1:
#            imLY=train_model.imageDownsample(imLY,srforest['sropts']['sf'],srforest['sropts']['downsample'])
            imLY=cv2.resize(imLY, None, fx=1/srforest['sropts']['sf'],fy=1/srforest['sropts']['sf'], interpolation=cv2.INTER_CUBIC)
            if len(imLCB)!=0 and len(imLCR)!=0:
#                imLCB=train_model.imageDownsample(imLCB,srforest['sropts']['sf'],srforest['sropts']['downsample'])
                imLCB=cv2.resize(imLCB, None, fx=1/srforest['sropts']['sf'],fy=1/srforest['sropts']['sf'], interpolation=cv2.INTER_CUBIC)
                imLCR=cv2.resize(imLCR, None, fx=1/srforest['sropts']['sf'],fy=1/srforest['sropts']['sf'], interpolation=cv2.INTER_CUBIC)
#                imLCR=train_model.imageDownsample(imLCR,srforest['sropts']['sf'],srforest['sropts']['downsample'])

#bicubic upsampling of the Y channel to generate the mid-res image
#        imMY = train_model.imresize(imLY,srforest['sropts']['sf'],srforest['sropts']['interpkernel'])
        imMY=cv2.resize(imLY, None, fx=srforest['sropts']['sf'],fy=srforest['sropts']['sf'], interpolation=cv2.INTER_CUBIC)
#generate super-resolution forest output
        imHYPred = applySRF(imMY,srforest,opts['Mhat'],opts['nthreads'])
        
        if isinstance(dataHigh,str) and len(dataHigh)!=0:
            imH=misc.imread(os.path.join(dataHigh,imlistHigh[i]))
        else:
            imH = dataHigh
        if len(imH)!=0:
#            prepare GT image
            imHYGI=train_model.imageTransformColor(imH)
            imHYGI=train_model.imageModcrop(imHYGI,srforest['sropts']['sf'])
#            check if rmborder is enough!
            rmBorder=(srforest['sropts']['patchBorder']+np.floor((srforest['sropts']['patchSizeLow']-srforest['sropts']['patchSizeHigh'])/2)).astype(int)
            imSize=imHYGI.shape
            imSize=imSize-2*rmBorder
            rmBorder = rmBorder+np.mod((imSize-srforest['sropts']['patchSizeHigh']),srforest['sropts']['patchStride'])
            if (opts['rmborder'])<rmBorder.any():
                raise ValueError('opts.rmBorder is set too small')
#            remove border for evaluation
            imHYPredEval=cropBorder(imHYPred,opts['rmborder'])
            imMYEval=cropBorder(imMY,opts['rmborder'])
            imHYGTEval=cropBorder(imHYGI,opts['rmborder'])
#            evaluate SRF & bicubic upsampling
            out.append({})
            out[i]['eval']={}
            out[i]['eval']['srf']={}
            out[i]['eval']['srf']['pnsr']=evaluateQuality(imHYGTEval,imHYPredEval)
            out[i]['eval']['bic']={}
            out[i]['eval']['bic']['pnsr']=evaluateQuality(imHYGTEval,imMYEval)
#        generate output image (RGB or grayscale) for SRF
        if len(imLCB)!=0 and len(imLCR)!=0:
            imHCB=train_model.imresize(imLCB,srforest['sropts']['sf'],srforest['sropts']['interpkernel'])
            imHCR=train_model.imresize(imLCR,srforest['sropts']['sf'],srforest['sropts']['interpkernel'])
            imHCB=imHCB.astype(np.float32)
            imHCR=imHCR.astype(np.float32)
            outimageYCBCR=cv2.merge([imHYPred,imHCB,imHCR])
            outimg=color.ycbcr2rgb(outimageYCBCR)
        else:
            outimg=255*imHYPred
        out[i]['im']=outimg
    return out
            

def evaluateQuality( imgt, impred ):
    imgt=imgt.astype(np.float64)
    impred=impred.astype(np.float64)
    pnsr=measure.compare_psnr(imgt,impred)
    return pnsr
            
def cropBorder( im, rmBorder ):
    imnoborder=im[rmBorder:-rmBorder,rmBorder:-rmBorder]
    return imnoborder
        
        
def applySRF( imMY, srforest, Mhat, nthreads ):
#    set some constants
    opts=srforest['sropts']
    tarBorder = opts['patchBorder']
    
#     extract patches and compute features
    patchesSrc=train_model.extractPatches(imMY,opts['patchSizeLow'],opts['patchSizeLow']-opts['patchStride'],opts['patchBorder'],opts['patchfeats'])
    patchesSrcPca = srforest['Vpca'].T.dot( patchesSrc)
    
# apply random regression forest
    if opts['useARF']==1:
        pass
    else:
        patchesTarPred=forestRegrApply(patchesSrcPca,patchesSrcPca,srforest['model'],Mhat,nthreads)
        patchestar=patchesTarPred[0]
        for i in range(Mhat-1):
            patchestar=patchestar+patchesTarPred[i+1]
        patchesTarPred=patchestar/len(patchesTarPred)
        
#add mid-res patches + patches predicted by SRF
    patchesMid = train_model.extractPatches(imMY,opts['patchSizeHigh'],opts['patchSizeHigh']-opts['patchStride'],tarBorder)
    patchesTarPred = patchesTarPred + patchesMid
    
#merge patches into the final output (i.e., average overlapping patches)
    a,b=np.shape(imMY)
    img_size=np.array([a,b])
    patchSizeHigh=srforest['sropts']['patchSizeHigh']
    grid=train_model.getSamplingGrid(img_size,patchSizeHigh,patchSizeHigh-opts['patchStride'],tarBorder,1)
    imout=overlap_add(patchesTarPred,img_size,grid,opts)
    
    return imout
        
#Image construction from overlapping patches
def overlap_add( patches, img_size, grid,opts):
    result=np.zeros(img_size,dtype=np.float32)
    weight=np.zeros(img_size)
    border=opts['patchBorder']
    patchsize=opts['patchSizeHigh']
    overlap=opts['patchSizeHigh']-opts['patchStride']
    skip=patchsize-overlap
    
    ref=result[border[0]:result.shape[0]-border[0],border[1]:result.shape[1]-border[1]]
    rew=weight[border[0]:weight.shape[0]-border[0],border[1]:weight.shape[1]-border[1]]
    i=0
    j=0
    for kk in range(grid.shape[2]):
        patch=patches[:,kk].reshape([grid.shape[0],grid.shape[1]],order='F')
#        for j in range(ref.shape[1]):
#        if (patchsize[1]+j*skip[1])>ref.shape[1]:
#            break
#            for i in range(ref.shape[0]):
        ref[i*skip[0]:patchsize[0]+i*skip[0],j*skip[1]:patchsize[1]+j*skip[1]]=\
        ref[i*skip[0]:patchsize[0]+i*skip[0],j*skip[1]:patchsize[1]+j*skip[1]]+patch
        rew[i*skip[0]:patchsize[0]+i*skip[0],j*skip[1]:patchsize[1]+j*skip[1]]=\
        rew[i*skip[0]:patchsize[0]+i*skip[0],j*skip[1]:patchsize[1]+j*skip[1]]+1
        i=i+1
        if (patchsize[0]+i*skip[0])>ref.shape[0]:
            i=0
            j=j+1
    result[border[0]:result.shape[0]-border[0],border[1]:result.shape[1]-border[1]]=ref
    weight[border[0]:weight.shape[0]-border[0],border[1]:weight.shape[1]-border[1]]=rew
    RW=ref/rew
    result[border[0]:result.shape[0]-border[0],border[1]:result.shape[1]-border[1]]=RW
    
    return result
        
def forestRegrApply( Xfeat, Xsrc, forest, Mhat, NCores ):
#    M=len(forest)
    nthreads = 1
#get the leaf node prediction type and check with the given Xsrc data
    Fs=forest[0]['leafinfo'][-1]['T'].shape[1]
    leafpredtype=forest[0]['leafinfo'][-1]['type']
    if leafpredtype==0:
        assert(Fs==1)
    elif leafpredtype==1:
        assert(Fs==Xsrc.shape[0]+1)
        Xsrc=np.row_stack((Xsrc,np.ones(Xsrc.shape[1])))
    elif leafpredtype==2:
        assert(Fs==Xsrc.shape[0]+1)
        Xsrc=np.vstack((Xsrc,np.ones(Xsrc.shape[1],Xsrc**2)))
#    iterate the trees
    myforest=forest[0:Mhat]
    node2leafids=[]
    treeleafs=[]
    for index in range(Mhat):
        node2leafids.append([])
        treeleafs.append([])
    for i in range(len(myforest)):
        tree=myforest[i]
        node2leafids[i]=np.zeros(tree['child'].shape,np.int32)
        for j in range(len(tree['child'])):
            if tree['child'][j]==0:
                node2leafids[i][j]=tree['leafinfo'][j]['id']
                treeleafs[i].append(tree['leafinfo'][j])
    
    XtarPred = forestRegrInference(Xfeat,Xsrc,myforest,node2leafids,treeleafs,leafpredtype,nthreads)
    
    return XtarPred


def  forestRegrInference(Xfeat,Xsrc,myforest,node2leafids,treeleafs,leafpredtype,nthreads):
    N=Xfeat.shape[1]
    opts={}
    opts['N1']=N
    opts['F1']=5
    opts['M']=10
    Xtarpred=[]
    for i in range(opts['M']):
        d=(np.arange(0,N)).T
        Xfeat1=Xfeat[:,d]
        Xsrc1=Xsrc[:,d]
        Xtarpred.append([])
        Xtarpred[i]=treeRegrApply(Xfeat1,Xsrc1,myforest[i],node2leafids[i],treeleafs[i],leafpredtype)
    
    return Xtarpred       
        
def treeRegrApply( Xfeat, Xsrc, mytree,node2leafid,treeleaf, leafpredtype):
    fids=mytree['fids']
    thrs=mytree['thrs']
    child=mytree['child']
    K=Xfeat.shape[1]
    Xtpred=np.zeros((81,K))
    for i in range(K):
        Xpatch=Xsrc[:,i]
        k=1
        while child[k-1,0]:
            if leafpredtype==1:
                if (Xpatch[fids[k-1,0]]-Xpatch[fids[k-1,1]])<thrs[k-1]:
                    k=child[k-1,0]
                else:
                    k=child[k-1,0]+1
            elif leafpredtype == 2: 
                if Xpatch[fids[k]]<thrs[k]:
                    k=child[k-1,0]
                else:
                    k=child[k-1,0]+1
        leafid=k
        node=node2leafid[leafid-1,0]
        leafT=treeleaf[node-1]['T']
#        n=leafT.shape[0]
        Xtpred[:,i]=leafT.dot(Xpatch)
    
    Xpred=Xtpred[:,0:K]
    
    return Xpred