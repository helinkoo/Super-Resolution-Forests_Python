# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 20:08:35 2017

@author: Administrator
"""
import os
import numpy as np
#from PIL import Image
import cv2
from scipy import misc
import forestRegrtrain
#import getPrmDflt as GP
from scipy import signal
import math
import sys

def srForestTrain( varargin ):
    dfs={ 'datapathHigh':[], 'datapathLow': [],  'sf':2,  
         'downsample':[],'patchSizeLow':np.array([6,6]),'patchSizeHigh':np.array([6,6]),
         'patchStride':np.array([2,2]), 'patchBorder':np.array([2,2]), 
         'nTrainPatches':0,  'nAddBaseScales':0,  'patchfeats':forestRegrtrain.forestRegrTrain(), 
         'interpkernel':'bicubic', 'pRegrForest':{},  
         'useARF':0,  'verbose':1  }
#    dfs['pRegrForest']=FR.forestRegrTrain()
    opts=getPrmDflt(varargin,dfs,1)
    if len(sys.argv)==0:
        srforest=opts
        return srforest
    
# check some parameter settings
    if opts['sf']<2:
        print('ValueError,sf should be >= 2')
    elif opts['patchSizeLow'][0]<3:
        print('ValueError,patchSizeLow >= 2')
    elif opts['patchSizeHigh'][0]<3:
        print('ValueError,patchSizeHigh >= 2')
    elif opts['nTrainPatches']<0:
        print('ValueError,nTrainPatches >= 0')
    elif opts['patchStride']>opts['patchSizeHigh'][0]:
        print('ValueError,Stride is too large for high-res patch size')
        
# extract source and target patches
    imlistLow = ' '
    nPatchesPerImg = 0

    #imlistHigh = list(open (opts['datapathHigh'],'rb+'))
    imlistHigh = opts['datapathHigh']
    #[i for i in os.listdir('.') if os.path.isfile(i) and os.path.splitext(i)[1]=='.bmp']
    listH = os.listdir(imlistHigh)  #列出文件夹下所有的目录与文件
    
    patchesSrcCell=np.zeros([1,len(listH)]) 
    patchesTarCell=np.zeros([1,len(listH)])
    
    for i in range(0,len(listH)):
    #for filename in os.listdir(r"./file"): 
        path = os.path.join(imlistHigh,listH[i])
        if os.path.isfile(path) and os.path.splitext(i)[1]=='.bmp':   #想对文件的操作
            if opts['verbose']==1:
               print('Extract patches from image %d/%d',i,len(listH))
            imL=[]
            imH =np.array(cv2.imread(listH[i],cv2.IMREAD_COLOR))
            if imH == None:                   #判断读入的img1是否为空，为空就继续下一轮循环
                continue
            baseScales=1
            patchesSrcCell[1,i],patchesTarCell[1,i] = extractPatchesFromImg(imH,imL,opts,baseScales)
            if nPatchesPerImg>0:
                pass
    patchesSrc=np.array([])
    patchesTar=np.array([])
    for j in range(0,len(listH)):
        patchesSrc=np.hstack((patchesSrc,np.array(patchesSrcCell[1,j])))
        patchesTar=np.hstack((patchesTar,np.array(patchesTarCell[1,j])))
    del patchesSrcCell,patchesTarCell
    if opts['verbose']==1:
        print('Extracted a total of %d patches from %d images\n'%patchesSrc.shape[1],len(listH))
    opts['pRegrForest']['N1']=patchesSrc.shape[1]
    
#reduce dimensionality of low-res patches (PCA)主成分分析法 降维
    if opts['verbose']==1:
        print('Applying PCA dim-reduction\n')
    meanVal=np.mean(patchesSrc,axis=0)
    newData=patchesSrc-meanVal
    covMat=np.cov(patchesSrc,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigVects=eigVects.cumsum(0)/sum(eigVects)
    k=0
    for i in range(0,len(eigVects)):
        if eigVects[i]>=(1e-3):
            k=k+1
    srforest={}
    srforest['Vpca']=eigVals[:,k:eigVals.shape[1]+1]
    del eigVals,eigVects,covMat
    print(' %d to %d dimensions\n'%srforest['Vpca'].shape[0],srforest['Vpca'].shape[1])
    patchesSrcPca=np.array([])
    patchesSrcPca=np.array(np.dot(srforest['Vpca'].T*patchesSrc))

#train the regression forest
    if opts['useARF']==1:
        if opts['verbose']==1:
            pass
    elif opts['useARF']==0:
        if opts['verbose']==1:
            print('Training alternating regression forest\n')
        srforest['model']=forestRegrtrain.forestRegrTrain(patchesSrcPca,patchesSrcPca,patchesTar,opts['pRegrForest'])
    srforest['sropts']=opts


def extractPatchesFromImg(imH,imL,opts,baseScales):
     baseScales=1
     srcPatches=[]
     tarPatches=[]
     
# down sample image(s)
     imH = imageDownsample(imH,baseScales,opts['downsample'])
     if len(imL)!=0:
         imL=imageDownsample(imL,baseScales,opts['downsample'])
     imHY=imageTransformColor(imH)
     imHY=imageModcrop(imHY,opts['sf'])
     if len(imL)==0:
        imLY=imageDownsample(imHY,opts['sf'],opts['downsample'])
     else:
         imLY=imageTransformColor(imL)
     imMY=imresize(imLY,opts['sf'],opts['interpkernel'])
     imTar=imHY-imMY
    
     AM=extractPatches(imMY,opts['patchSizeLow'],opts['patchSizeLow']-opts['patchStride'],opts['patchBorder'],opts['patchfeats'])
     srcPatches=np.array(srcPatches.append[AM])
     tarBorder=opts['patchBorder']+math.floor((opts['patchSizeLow']-opts['patchSizeHigh'])/2)
     TM=extractPatches(imTar,opts['patchSizeHigh'],opts['patchSizeHigh']-opts['patchStride'],tarBorder)
     tarPatches=np.array(tarPatches.append[TM])
     
     return srcPatches,tarPatches


def imageDownsample(imH,sf,varargin):
    dfs={ 'kernel':'bicubic',  'sigma': .4  }
    
    opTs=getPrmDflt(varargin,dfs,1)
    
    if opTs['kernel']=='bicubic':
        imL =misc.imresize(imH,1/sf,'bicubic')
    elif opTs['kernel']=='Gaussian':
        np.error('Not implemented yet!')
    else:
        np.error('Unknown kernel')
        
    return imL
        
        
def imresize(im,sf,interpkernel):
 #""" 使用PIL 对象重新定义图像数组的大小""" 
    if interpkernel == 'bicubic':
        pil_im = cv2.resize(im, None, fx=sf,fy=sf, interpolation=cv2.INTER_CUBIC)
        return pil_im
    else:
        raise np.error('error')


def imageModcrop(imHY,modfactor):
    imh,imw=np.shape(imHY)
    imh=imh-imh%modfactor
    imw=imw-imw%modfactor
    imcrop=imHY[0:imh+1,0:imw+1]
    return imcrop

#extractPatches Extracts patches from an image.  
def extractPatches( im, patchsize, overlap, border, varargin ):
    dfs={'type':'none','filters':[]}
    opts=getPrmDflt(varargin,dfs,1)
    a,b,c=np.shape(im)
    if c!=1:
        raise np.error('im should have only a single channel')
    if patchsize<3:
        raise np.error('patchsize shoud be >= 3')
    grid = getSamplingGrid(np.array(np.shape(im)),patchsize,overlap,border,1)
    if opts['type']=='none':
        f=im[grid]
        patches=f.reshape(f.shape[0]*f.shape[1],f.shape[2])
    elif opts['type']=='fliters':
        a,b=np.shape(opts['fliters'])
        feature_size=patchsize[0]*patchsize[1]*opts['fliters'].size
        patches=np.zeros([feature_size,grid.shape[2]])
        for i in opts['fliters'].size:
            f=signal.convolve2d(im,opts['filters'][i],mode='same')
            f=f[grid]
            f=f.reshape(f.shape[0]*f.shape[1],f.shape[2])
            patches[(i-1)*f.shape[0]:f.shape[0]+(i-1)*f.shape[0],:]=f
    else:
        raise np.error('UNknown featuers')
    return patches
    
#getSamplingGrid returns a grid to easily extract patches from an image    
def getSamplingGrid( img_size, patch_size, overlap, border, scale ):
    patch_size=np.array(patch_size)*scale
    overlap=np.array(overlap)*scale
    border=np.array(border)*scale
    
    X=img_size[0]
    Y=img_size[1]
    index=np.arange(1,X*Y+1).reshape(Y,X)
    index=index.T
    grid=index[0:patch_size[0],0:patch_size[1]]-1
#Compute offsets for grid's displacement.
    skip=patch_size-overlap
    offest=index[border[0]:img_size[0]-patch_size[0]-border[0]:skip[0],
                 border[1]:img_size[1]-patch_size[1]-border[1]:skip[1]]
    #L,T=np.shape(offest)
    offest=offest.reshape(1,1,offest.size)
#Prepare 3D grid - should be used as: sampled_img = img(grid)
    grid=np.tile(grid,[1,1,offest.size])+np.tile(offest,[patch_size[0],patch_size[1],1])
    return grid

def imageTransformColor(im):
     x,y,z=np.shape(im)
     if z==3:
         img=im.convert("YCbCr")
#     imY = img[:,:,0]
#     imCB= img[:,:,1]
#     imCR = img[:,:,2]
         imY,imCB,imCR = img.split()
# 怎么把图像转化为单精度的？
         imHY=np.mat(imY)*(1/255)
     return imHY
 
def getPrmDflt( prm, dfs, checkExtra=1 ):
#    if (len(dfs)%2 == 1):
#        raise ValueError('odd number of default parameters')
    if isinstance(prm,dict):
        pass
#    if (len(dfs)%2 == 1):
#        raise ValueError('odd number of default parameters')
    prmField=[]
    prmVal=[]
    for key,values in  prm.items():   
        prmField.append(key)
        prmVal.append(values)
# get and update default values using quick for loop
    dfsField=[]
    dfsVal=[]
	#radiansdict.update(dict2)：把字典dict2的键/值对更新到dict里
    for key,values in  dfs.items():   
        dfsField.append(key)
        dfsVal.append(values)
    if checkExtra==1:
        for index in range(len(prmField)):
            if (prmField[index]==dfsField[index]):
                j=index
                dfsVal[j] = prmVal[index]
#check for missing values
    pass
#set output
#    dfs.keys()=dfsField
#    dfs.values()=dfsVal
    
    return dfs
def getPrmDflt1(dfs, checkExtra=1,prm=None):
#    if (len(dfs)%2 == 1):
    if len(prm)==0:
        return dfs
#        raise ValueError('odd number of default parameters')
    else:
        if isinstance(prm,dict):
            pass
        dfs.update(prm)
        '''
        prmField=[]
        prmVal=[]
        for key,values in  prm.items():   
            prmField.append(key)
            prmVal.append(values)
        # get and update default values using quick for loop
        dfsField=[]
        dfsVal=[]
        for key,values in  dfs.items():   
            dfsField.append(key)
            dfsVal.append(values)
        if checkExtra==1:
            for index in range(len(prmField)):
                if (prmField[index]==dfsField[index]):
                    j=index
#                    dfsVal[j] = prmVal[index]
                    dfs['']
        #check for missing values
        pass
        #set output
        #    dfs.keys()=dfsField
        #    dfs.values()=dfsVal
       ''' 
        return dfs
