# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:49:04 2018

@author: Administrator
""" 
import numpy as np

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