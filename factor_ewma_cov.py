# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:11:10 2018

@author: Hanbrendan
"""
import pandas as pd
import numpy as np

"""因子ewma方差计算"""
def ewma_cov_matirx(df,window,time,factorlist):
    def weightcreate(x,num):
        weight = (0.5)**(x)
        weightlist = [(weight**(num-i))**2 for i in range(num)]
        return weightlist
    ewmaweight = weightcreate(1/63,window)
    if type(df) == str:
        factor_ret = pd.read_csv(df,parse_dates=[str(time)])
    elif type(df) == pd.DataFrame:
        factor_ret = df
    try:
        del factor_ret['Unnamed: 0']
    except:
        pass
    try:
        factor_ret[str(time)] = pd.to_datetime(factor_ret[str(time)])
    except:
        pass
    factorlist = [str(time)] + factorlist
    factorcov = factor_ret[factorlist] 
    factorcov = factorcov.sort_values(str(time))
    def cov_matrix(df):
        result = pd.DataFrame()
        for i in range(window,len(df)):
            temp = df.iloc[(i-window):i,1:]
            tryd = pd.DataFrame([temp.iloc[:,i] * ewmaweight for i in range(len(list(temp.columns)))])
            res = pd.DataFrame(np.cov(np.array(tryd)))
            #pd.DataFrame(np.cov(np.array(temp.T)))
            res.columns = list(df.columns[1:])
            res['Trddt'] = df.iloc[i,0]
            result = pd.concat([result,res],axis=0)
        return(result)
    factorcov = cov_matrix(factorcov)
    return factorcov