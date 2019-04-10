# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 02:12:47 2017

@author: HeLin
"""

import os
from scipy import io
import numpy as np 
import test_model
import train_model
import pickle


if __name__ == '__main__':
    scopts={}
    scopts['datapathHigh'] = 'data\\train'
    scopts['datapathLow'] = ''
    scopts['sf'] = 3   #upscaling factor
    scopts['downsample']={}
    scopts['downsample']['kernel'] = 'bicubic'
    scopts['downsample']['sigma'] = 0
    scopts['patchSizeLow'] = np.array([3,3]) * scopts['sf']
    scopts['patchSizeHigh'] = np.array([3,3])* scopts['sf'] 
    scopts['patchStride'] =  np.array([1,1])* scopts['sf'] 
    scopts['patchBorder'] =  np.array([1,1])* scopts['sf'] 
    scopts['nTrainPatches'] = 0
    scopts['nAddBaseScales'] = 0
    scopts['patchfeats']={}
    scopts['patchfeats']['type'] = 'filters'
    O =  np.zeros([scopts['sf']-1])
    G =  np.mat(np.hstack([1,O,-1]))  # Gradient % O = [0,0]
    L =  np.mat(np.hstack([1,O,-2,O,1]))/2  # Laplacian
    scopts['patchfeats']['filters'] = list([G, G.T, L, L.T])
    scopts['interpkernel'] = 'bicubic'
    scopts['pRegrForest'] = train_model.forestRegrTrain() 
    scopts['pRegrForest']['M'] = 10
    scopts['pRegrForest']['maxDepth'] = 15
    scopts['pRegrForest']['nodesubsample'] = 512
    scopts['pRegrForest']['verbose'] = 1
    scopts['pRegrForest']['usepf'] = 1
    scopts['useARF'] = 0   #requires longer training times!
    
    # path to the model file
    srforestPath = 'models\\'
    srforestFNm = ('srf_sf-%d_T-%02d_ARF-%d.pkl'%(scopts['sf'], scopts['pRegrForest']['M'],scopts['useARF']))
    srforestFNm = os.path.join(srforestPath , srforestFNm)
    
    # path to test images
    datapathTestHigh = 'data\\test_Set5'
    datapathTestLow = ''
    
    method=2
    ## train the super-resolution forest
    if os.path.exists(srforestFNm): #判断是否存在模型
        print('Loading super-resolution forest\n')
        # srforest = srForestLoad(srforestFNm)
#load .mat file
        if method==1:
            forest=io.loadmat(srforestFNm)
            srforest={}
            srforest['model']=forest['model']
            srforest['Vpca']=forest['Vpca']
            srforest['sropts']=forest['sropts']
#load .pkl file
        elif method==2:
            srforest=pickle.load(open(srforestFNm,'rb'))
        
    else:
        print('Training super-resolution forest\n')
        srforest = train_model.srForestTrain( scopts )
#save as .mat file
        if method==1:
            io.savemat(srforestFNm,srforest)
#save as .pkl file
        elif method==2:
            pickle.dump(srforest,open(srforestFNm,'wb'),protocol=4)

# testing the learned model       
    outstats = test_model.srForestApply(datapathTestLow,datapathTestHigh,srforest,{'rmborder':3})
    
#visualize some results
    print('\nBicubic Upsampling (x%d): \n'%scopts['sf'])
    psnr_m=np.zeros((len(outstats),1))
    for i in range(len(outstats)):
        psnr_m[i]=outstats[i]['eval']['bic']['pnsr']
        print('Img %d/%d: psnr = %.2f dB\n'%(i,len(outstats),psnr_m(i)))
    print('===\nMean PSNR = %.2f dB\n'% np.mean(psnr_m))
    print('\nSRF Upsampling (x%d): \n'%scopts['sf'])
    psnr_m=np.zeros((len(outstats),1))
    for i in range(len(outstats)):
        psnr_m[i]=outstats[i]['eval']['srf']['pnsr']
        print('Img %d/%d: psnr = %.2f dB\n'%(i,len(outstats),psnr_m(i)))
    print('===\nMean PSNR = %.2f dB\n'% np.mean(psnr_m))
  