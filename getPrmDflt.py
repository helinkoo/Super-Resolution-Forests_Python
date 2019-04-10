# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:14:25 2018

@author: Administrator
"""
def getPrmflt( prm, dfs, checkExtra=1 ):
    if (len(dfs)%2 == 1):
        raise ValueError('odd number of default parameters')
    if isinstance(prm,dict):
        pass
    if (len(dfs)%2 == 1):
        raise ValueError('odd number of default parameters')
    prmField=prm.keys()
    prmVal=prm.values()
# get and update default values using quick for loop
    dfsField=dfs.keys()
    dfsVal=dfs.values()
    if checkExtra==1:
        for index in range(len(prmField)):
            if (prmField[index]==dfsField[index]):
                j=index
                dfsVal[j] = prmVal[index]
#check for missing values

#set output
#    dfs.keys=dfsField
#    dfs.values=dfsVal
    return dfs