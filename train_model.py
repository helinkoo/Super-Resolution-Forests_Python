# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:53:39 2018

@author: HeLin
"""

import os
import numpy as np 
#from PIL import Image
import cv2
from scipy import misc
from scipy import signal
import math
import warnings
import gc
from skimage import color
#from scipy import io
from sklearn.decomposition import PCA
#from skimage import transform
#import matplotlib.image as mpimg
      
      
def srForestTrain( varargin ):
    dfs={ 'datapathHigh':[], 'datapathLow': [],  'sf':2,  
         'downsample':[],'patchSizeLow':np.array([6,6]),'patchSizeHigh':np.array([6,6]),
         'patchStride':np.array([2,2]), 'patchBorder':np.array([2,2]), 
         'nTrainPatches':0,  'nAddBaseScales':0,  'patchfeats':[], 
         'interpkernel':'bicubic', 'pRegrForest':forestRegrTrain(),  
         'useARF':0,  'verbose':1 }
#    dfs['pRegrForest']=FR.forestRegrTrain()
    opts=getPrmDflt(dfs,1,varargin)
    if len(varargin)==0:
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
    elif (opts['patchStride']).any()>(opts['patchSizeHigh']).any():
        print('ValueError,Stride is too large for high-res patch size')       
# extract source and target patches
    imlistLow = ' '
    nPatchesPerImg = 0
    #imlistHigh = list(open (opts['datapathHigh'],'rb+'))
    imlistHigh = opts['datapathHigh']
    #[i for i in os.listdir('.') if os.path.isfile(i) and os.path.splitext(i)[1]=='.bmp']
    listH = os.listdir(imlistHigh)  #列出文件夹下所有的目录与文件
    if len(opts['datapathLow'])!=0:
        imlistLow=os.listdir( opts['datapathLow'])        
    patchesSrcCell=[] 
    patchesTarCell=[]
    for i in range(0,len(listH)):
        patchesSrcCell.append([])
        patchesTarCell.append([])
        path = os.path.join(imlistHigh,listH[i])
#        if os.path.isfile(path) and os.path.splitext(i)[1]=='.bmp':   #想对文件的操作
        if opts['verbose']==1:
           print('Extract patches from image %d/%d'%(i+1,len(listH)))
        imL=[]
#        imH =np.array(cv2.imread(listH[i],cv2.IMREAD_COLOR))
#        imH=cv2.imread(path)
        imH=misc.imread(path)
#        cv2.namedWindow("Image")
#        cv2.imshow("Image", imH)
#        cv2.waitKey (0)
#        cv.destroyAllWindows()
        if imH.any() == None:                   #判断读入的img1是否为空，为空就继续下一轮循环
            continue
        baseScales=1
        patchesSrcCell[i],patchesTarCell[i] = extractPatchesFromImg(imH,imL,opts,baseScales)
        if nPatchesPerImg>0:
            pass
    patchesSrc=patchesSrcCell[0]
    patchesTar=patchesTarCell[0]
    for j in range(len(listH)-1):
        patchesSrc=np.hstack((patchesSrc,patchesSrcCell[j+1]))
        patchesTar=np.hstack((patchesTar,patchesTarCell[j+1]))
    del patchesSrcCell
    gc.collect()
    del patchesTarCell
    gc.collect()
    if opts['verbose']==1:
        print('Extracted a total of %d patches from %d images\n'%(patchesSrc.shape[1],len(listH)))
    opts['pRegrForest']['N1']=patchesSrc.shape[1]    
#reduce dimensionality of low-res patches (PCA)主成分分析法 降维
    if opts['verbose']==1:
        print('Applying PCA dim-reduction\n')
#    meanVal=np.mean(patchesSrc,axis=0)
#    newData=patchesSrc-meanVal
    srforest={}
    pca = PCA(n_components=30) 
    datapca=pca.fit_transform(patchesSrc)
    srforest['Vpca']=datapca

    print(' %d to %d dimensions\n'%(srforest['Vpca'].shape[0],srforest['Vpca'].shape[1]))
    patchesSrcPca=np.array([])
    patchesSrcPca=np.array(np.dot(srforest['Vpca'].T,patchesSrc))
#train the regression forest
    if opts['useARF']==1:
        if opts['verbose']==1:
            pass
    elif opts['useARF']==0:
        if opts['verbose']==1:
            print('Training regression forest\n')
        srforest['model']=forestRegrTrain(patchesSrcPca,patchesSrcPca,patchesTar,opts['pRegrForest'])
    srforest['sropts']=opts
    
    return srforest


def extractPatchesFromImg(imH,imL,opts,baseScales):
     baseScales=1
     srcPatches=[]
     tarPatches=[]     
# downsample image(s)
     imH = imageDownsample(imH,baseScales,opts['downsample'])
     if len(imL)!=0:
         imL=imageDownsample(imL,baseScales,opts['downsample'])
     imHY=imageTransformColor(imH)
     imHY=imageModcrop(imHY,opts['sf'])
     if len(imL)==0:
        imLY=imageDownsample(imHY,opts['sf'],opts['downsample'])
     else:
        imLY=imageTransformColor(imL)
#     size=imHY.shape
     imMY=imresize(imLY,opts['sf'],opts['interpkernel'])
#     imMY=imMY/255
     imTar=imHY-imMY    
     Spatches=extractPatches(imMY,opts['patchSizeLow'],opts['patchSizeLow']-opts['patchStride'],opts['patchBorder'],opts['patchfeats'])
     srcPatches=Spatches
     tarBorder=opts['patchBorder']
     Tpatches=extractPatches(imTar,opts['patchSizeHigh'],opts['patchSizeHigh']-opts['patchStride'],tarBorder)
     tarPatches=Tpatches
     
     return srcPatches,tarPatches


def imageDownsample(imH,sf,varargin):
    dfs={ 'kernel':'bicubic',  'sigma': .4  }    
    opts=getPrmDflt(dfs,1,varargin)   
    if opts['kernel']=='bicubic':
        if sf==1:
            imL =misc.imresize(imH,1/sf,'bicubic')
#            imL =transform.resize(imH,(imH.shape[0]/sf,imH.shape[1]/sf))
#            imL=cv2.resize(imH, None, fx=1/sf,fy=1/sf, interpolation=cv2.INTER_CUBIC)
        elif sf!=1:  
            imL =misc.imresize(imH,1/sf,'bicubic')
#            imL =transform.resize(imH,(imH.shape[0]/sf,imH.shape[1]/sf))
#            imLY=cv2.resize(imLY, None, fx=1/sf,fy=1/sf, interpolation=cv2.INTER_CUBIC)
            imL=imL/255
    elif opts['kernel']=='Gaussian':
        ValueError('Not implemented yet!')
    else:
        ValueError('Unknown kernel')
        
    return imL
        
        
def imresize(im,sf,interpkernel):
 #""" 使用PIL 对象重新定义图像数组的大小""" 
    if interpkernel == 'bicubic':
        pil_im = cv2.resize(im, None, fx=sf,fy=sf, interpolation=cv2.INTER_CUBIC)
#        pil_im = cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)
        return pil_im
    else:
        raise ValueError('error')


def imageModcrop(imHY,modfactor):
    imh,imw=np.shape(imHY)
    imh=imh-imh%modfactor
    imw=imw-imw%modfactor
    imcrop=imHY[0:imh,0:imw]
    return imcrop

'''
def imageModcrop(image, modfactor):
	if image.shape[2] == 1:
		size = image.shape
		size -= np.mod(size, modfactor)
		image = image[0:size[0], 0:size[1]]
	else:
		size = image.shape[0:2]
		size -= np.mod(size, modfactor)
		image = image[0:size[0], 0:size[1], 0]
	return image
'''

def modcrop_color(image, modfactor):
	size = image.shape[0:2]
	size -= np.mod(size, modfactor)
	image = image[0:size[0], 0:size[1], :]
	return image

#extractPatches Extracts patches from an image.  
def extractPatches( im, patchsize, overlap, border, varargin={} ):
    dfs={'type':'none','filters':[]}
    opts=getPrmDflt(dfs,1,varargin)
    size=np.shape(im)
    skip=patchsize-overlap
    if len(size)==3:
        raise ValueError('im should have only a single channel')
    if patchsize[0]<3 or patchsize[1]<3:
        raise ValueError('patchsize shoud be >= 3')
    grid = getSamplingGrid(size,patchsize,overlap,border,1)
    if opts['type']=='none':
        patches=np.zeros([patchsize[0]*patchsize[1],grid.shape[2]])
        ref=im[border[0]:im.shape[0]-border[0],border[1]:im.shape[1]-border[1]]
        (a,b,c)=grid.shape
        kk=0
        pf=np.zeros([a,b,c])
        for j in range(ref.shape[1]):
            if (patchsize[1]+j*skip[1])>ref.shape[1]:
                break
            for i in range(ref.shape[0]):
                if (patchsize[0]+i*skip[0])>ref.shape[0]:
                    break
                pf[:,:,kk]=ref[i*skip[0]:patchsize[0]+i*skip[0],j*skip[1]:patchsize[1]+j*skip[1]]
                kk=kk+1
        f=pf
        patches=f.reshape(f.shape[0]*f.shape[1],f.shape[2],order='F')
        return patches
    elif opts['type']=='filters':
        feature_size=patchsize[0]*patchsize[1]*len(opts['filters'])
        patches=np.zeros([feature_size,grid.shape[2]])  
        for index in range(len(opts['filters'])):
            f=signal.convolve2d(im,opts['filters'][index],mode='same')
#            f=f[grid]
            ref=f[border[0]:f.shape[0]-border[0],border[1]:f.shape[1]-border[1]]
            (a,b,c)=grid.shape     
            kk=0
            pf=np.zeros([a,b,c])
            for j in range(ref.shape[1]):
                if (patchsize[1]+j*skip[1])>ref.shape[1]:
                    break
                for i in range(ref.shape[0]):
                    if (patchsize[0]+i*skip[0])>ref.shape[0]:
                        break
                    pf[:,:,kk]=ref[i*skip[0]:patchsize[0]+i*skip[0],j*skip[1]:patchsize[1]+j*skip[1]]
                    kk=kk+1
            f=pf     
            f=f.reshape(f.shape[0]*f.shape[1],f.shape[2],order='F')
            patches[index*f.shape[0]:f.shape[0]+index*f.shape[0],:]=f
#            patches[(1:f.shape[0])+(index-1)*f.shape[0],:]=f
        return patches
    else:
        raise ValueError('UNknown featuers')

    
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
    offest=index[border[0]:img_size[0]-patch_size[0]-border[0]+1:skip[0],border[1]:img_size[1]-patch_size[1]-border[1]+1:skip[1]]         
    #L,T=np.shape(offest)
    offest=offest.T.reshape(1,1,offest.size)
#Prepare 3D grid - should be used as: sampled_img = img(grid)
    grid=grid.reshape(9,9,1)
    grid=np.tile(grid,(1,1,offest.shape[2]))+np.tile(offest,(patch_size[0],patch_size[1],1))
    return grid

def imageTransformColor(im):
     x,y,z=np.shape(im)
     if z==3:
         img=color.rgb2ycbcr(im)
         imG=np.around(img)
         imY,imCR,imCB = cv2.split(imG)
         imHY=imY/255
         return imHY
     
def imageTransformColorT(im):
     x,y,z=np.shape(im)
     if z==3:
         img=color.rgb2ycbcr(im)
         imG=np.around(img)
         imY,imCR,imCB = cv2.split(imG)
         imY=imY/255
         imCR=imCR/255
         imCB=imCB/255
         return imY,imCB,imCR
'''
         imY,imCR,imCB = cv2.split(img)
# 怎么把图像转化为单精度的？
#         imHY=np.mat(imY)*(1/255)
         imHY = im2single(imY)
         imCB = im2single(imCB)
         imCR = im2single(imCR)
'''

def im2single(im):
	info = np.iinfo(im.dtype) # Get the data type of the input image
	return im.astype(np.float32) / info.max


def getPrmDflt(dfs, checkExtra=1,prm=None):
#    if (len(dfs)%2 == 1):
    if len(prm)==0:
        return dfs
#        raise ValueError('odd number of default parameters')
    else:
        if isinstance(prm,dict):
            pass
        dfs.update(prm)
        return dfs


def forestRegrTrain( *args ):
    if len(args)!=0:
        Xfeat=args[0]
        Xsrc=args[1]
        Xtar=args[2]
        varargin=args[3]
    else:
        varargin={}
    dfs={ 'M':1,  'minChild':64,  'minCount':128,  'N1':[],  'F1':5,  'F2':5,
         'maxDepth':64,  'fWts':[],  'splitfuntype':'pair', 
      'nodesubsample':1000,  'splitevaltype':'variance', 
      'lambda':0.01,  'estimatelambda':0,  'kappa':1,  'leaflearntype':'linear',
      'usepf':0,  'verbose':0 }
    opts=getPrmDflt(dfs,1,varargin)
    if len(args)==0:
        forest=opts
        return forest
    
    Ff,N=np.shape(Xfeat)
    Ncheck=Xsrc.shape[1]
    assert(N==Ncheck)
    Ncheck=Xtar.shape[1]
    assert(N==Ncheck)
    if opts['N1']==0:
        opts['N1']=round(N*0.75)
        opts['N1']=min(N,opts['N1'])
    if opts['F1']==0:
        opts['F1']=round(math.sqrt(Ff))
        opts['F1']=min(Ff,opts['F1'])
    if opts['F2']<0:
        raise ValueError('F2 should be > -1')
    if opts['fWts']==[]:
        opts['fWts']=np.ones((1,Ff))
        opts['fWts']=opts['fWts']/sum(sum(opts['fWts']))
    if opts['nodesubsample']<opts['minChild']*2:
        raise ValueError('nodesubsample < 2*minChild')
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
               d=(np.arange(0,N)).T
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
        raise ValueError('Unknown splitfunction type')
    child=np.zeros((K,1),np.int32)
    count=np.zeros((K,1),np.int32)
    depth=np.zeros((K,1),np.int32)
    leaf_info={'T':[],'type':-1,'id':-1}
#     leafinfo=leafinfo(np.ones((K,1)))
    "map函数的使用？"
    leafinfo=[]
    for i in range(K):
        leafinfo.append(leaf_info)
    dids=[]
    for j in range(K):
        dids.append([])
    dids[0]=np.arange(N)
    k=0
    K=2
    msgnodestep=200
# train the tree   
    while (k+1<K):
        dids1=dids[k]
        count[k]=len(dids1)
        XfeatNode=Xfeat[:,dids1]
        XsrcNode=Xsrc[:,dids1]
        XtarNode=Xtar[:,dids1]
        if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
            print('Node %d, depth %d, %d samples () '%(k,depth[k],count[k]))
        if count[k]<=opts['minCount'] or depth[k]>opts['maxDepth'] or count[k]<(2*opts['minChild']):
            if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
                print('becomes a leaf (stop criterion active)\n')
            leafinfo[k]=createLeaf(XsrcNode,XtarNode,opts['leaflearntype'],opts['lambda'],opts['estimatelambda'])
            k=k+1
            continue
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
            raise ValueError('Unknown splitfunction type')
    #subsample the data for splitfunction node optimization 随机选样本
        if opts['nodesubsample']>0 and opts['nodesubsample']<count[k]:
            JJ=count[k][0]
            randinds=np.random.permutation(JJ)
            randinds=randinds[0:opts['nodesubsample']]
            respSub=resp[:,randinds]
            XsrcSub=XsrcNode[:,randinds]
            XtarSub=XtarNode[:,randinds]
        else:
            respSub = resp
            XsrcSub = XsrcNode
            XtarSub = XtarNode
        if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
            print('subsmpl = %07d/%07d () '%(respSub.shape[1],resp.shape[1]))
    #find best splitting function and corresponding threshold寻找最优的节点分裂函数和相对应的阈值
        [fid,thr,rerr]=findSplitAndThresh(respSub,XsrcSub,XtarSub,opts['F2'],opts['splitevaltype'],opts['lambda'],opts['minChild'],opts['kappa'])
    #check validity of the splitting function
        validsplit=0
        left=resp[fid,:]<thr
        count0=np.count_nonzero(left)
        fid=fids1[:,fid]
        if rerr!=np.inf and count0>=opts['minChild'] and (count[k][0]-count0)>=opts['minChild']:
            validsplit=1
        if validsplit==1:
            child[k]=K
            fids[k,:]=fid
            thrs[k]=thr
            dids[K-1]=dids1[left]
            right=~left
            dids[K]=dids1[right]
            depth[K-1:K+1]=depth[k][0]+1
            K=K+2
            dids[k]=[]
            k=k+1
            if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
                print('valid split (loss=%.6f)\n'%(rerr)) 
        else:
            if opts['verbose']==1 and (k%msgnodestep==0 or k==1):
                print('invalid split -> leaf\n')
            leafinfo[k]=createLeaf(XsrcNode,XtarNode,opts['leaflearntype'],opts['lambda'],opts['estimatelambda'])
            k=k+1
    K=K-1
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
    M=prob.shape[1]
    assert(N<=M)
    if N==M:
        pass
#    assert(prob.any()==prob[0])
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
            tthrs[0:-1]=np.random.rand(5,1)*0.95*(respmax-respmin)+respmin
            tthrs[-1]=np.median(resp[s,:])
        for t in range(len(tthrs)):
            tthr=tthrs[t][0]
            left=resp[s,:]<tthr
            right=~left
            nl=len(np.nonzero(left)[0])
            nr=len(np.nonzero(right)[0])
            '''
            nl=0
            nr=0
            for i in range(len(resp)):
                if left[i]==1:
                    nl=nl+1
                else:
                    nr=nr+1
            '''
#            left=left.astype('int')
#            right=right.astype('int')
            if nl<minChild or nr<minChild:
                continue
#  mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:] 
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
                raise ValueError('Unknown split evaluation type')
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
        raise ValueError('Unknown leaf node prediction type')
    leaf={'T':[],'type':-1,'id':-1}
    leaf['T']=I
    leaf['type']=predmodeltype
    leaf['id']=-1
    
    return leaf
    

def estimateLambda(matinv):
    pass

