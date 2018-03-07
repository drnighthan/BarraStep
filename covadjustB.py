# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:02:47 2018

@author: Hanbrendan
"""
import pandas as pd
import numpy as np

def B_adjustment(covm,facret,time,window):
    if type(covm) == str:
        covm = pd.read_csv(covm,parse_dates=[str(time)])
    elif type(covm)  == pd.DataFrame:
        covm[str(time)] = pd.to_datetime(covm[str(time)])
    if type(facret) == str:
        facret = pd.read_csv(facret,parse_dates=[str(time)])
    elif type(facret)  == pd.DataFrame:
        facret[str(time)] = pd.to_datetime(facret[str(time)])
    try:
        del covm['Unnamed: 0']
        del facret['Unnamed: 0']
    except:
        pass
    datelist = list(covm[str(time)].drop_duplicates())
    datelist2 = list(facret[str(time)].drop_duplicates())
    if len(datelist) >= len(datelist2):
        datelist = datelist2
    b2 = pd.DataFrame()
    for i in datelist:
        b2temp = pd.DataFrame()
        temp = covm[covm[str(time)] == i]
        temp = temp.set_index(str(time))
        temp2 = pd.DataFrame(np.diag(temp)).T
        temp2.columns = temp.columns.map(lambda x : str(x)+'_cov')
        tempret = pd.DataFrame(facret[facret[str(time)]==i][list(temp.columns)]).reset_index(drop=True)
        temp3 = pd.concat([temp2,tempret],axis=1)
        b2temp['B2'] = (1/len(temp.columns) * sum([temp3[i]**2/temp3[str(i)+'_cov'] for i in list(temp.columns)]))**0.5
        b2temp[str(time)] = i
        b2 = pd.concat([b2,b2temp],axis=0)
    def weightcreate(x,num):
        weight = (0.5)**(x)
        weightlist = [(weight**(num-i))**2 for i in range(num)]
        return weightlist
    ewmaweight = weightcreate(1/23,window)
    b2['lambda'] = b2['B2'].rolling(window).apply(lambda x : np.dot(x**2,ewmaweight)**0.5)
    factor_ret = pd.merge(facret,b2,on=[str(time)])
    factor_ret['B'] = factor_ret['B2'].map(lambda x:x**0.5)
    factor_ret = factor_ret.dropna()
    factor_ret.loc[factor_ret['B'] <1, 'lambda'] = 1
    factorcov_lambda = pd.DataFrame()
    datelist = list(factor_ret[str(time)].drop_duplicates())
    for i in datelist:
        lambda2 = factor_ret[factor_ret[str(time)]==i]['lambda'].astype(float)**2
        temp = covm[covm[str(time)]==i]
        del temp[str(time)]
        temp = temp.multiply(np.float(lambda2))
        temp[str(time)] =i
        factorcov_lambda = pd.concat([factorcov_lambda,temp],axis=0)
    return(factorcov_lambda)