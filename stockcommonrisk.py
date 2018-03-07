# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:16:07 2018

@author: Hanbrendan
"""
'''计算个股的因子部分的风险'''

import pandas  as pd
import numpy as np
def risk(factor_ret,ret_factorloading,factorlist,factorcov_lambda,time,stock,retname,window):
    if type(factor_ret) == str:
        ret_factorloading = pd.read_csv(ret_factorloading,parse_date=[str(time)])
    if type(factor_ret) == pd.DataFrame:
        factor_ret[str(time)] = pd.to_datetime(factor_ret[str(time)])
    if type(ret_factorloading) == str:
        ret_factorloading = pd.read_csv(ret_factorloading,parse_date=[str(time)])
    if type(ret_factorloading) == pd.DataFrame:
        ret_factorloading[str(time)] = pd.to_datetime(ret_factorloading[str(time)])
    try:
        del ret_factorloading['Unnamed: 0']
        del factor_ret['Unnamed: 0']
    except:
        pass
    factorlist2 = [str(time)] + factorlist
    factor_ret = factor_ret[factorlist2]
    factorliststock = [str(stock)] + factorlist2
    factorloading = ret_factorloading[factorliststock]
    datelist = list(factorcov_lambda[str(time)].drop_duplicates())
    commonriskresult = dict()
    for i in datelist:
        tempfactorloading = factorloading[factorloading[str(time)] == i]
        tempfactorcov = factorcov_lambda[factorcov_lambda[str(time)] == i]
        del tempfactorcov[str(time)]
        faloadtemp = tempfactorloading[tempfactorcov.columns]
        tempres = pd.DataFrame(np.dot(np.dot(faloadtemp,tempfactorcov),faloadtemp.T))
        tempres.columns = list(tempfactorloading[str(stock)])
        tempres['Stkcd'] = list(tempfactorloading[str(stock)])
        commonriskresult[i] = tempres
    idriskresult = pd.merge(ret_factorloading,factor_ret,on='Trddt')
    print(idriskresult)
    temp = pd.DataFrame([idriskresult[str(i)+'_x'] * idriskresult[str(i)+'_y'] for i in factorlist]).T
    temp = temp.sum(axis=1)
    idriskresult['alpha'] = idriskresult[str(retname)] - temp
    idriskresult = idriskresult[['Stkcd','Trddt','alpha']]
    def weightcreate(x,num):
        weight = (0.5)**(x)
        weightlist = [(weight**(num-i))**2 for i in range(num)]
        return weightlist
    ewmaweight = weightcreate(1/63,window)
    def idrisk(df):
        df = df.sort_values(str(time))
        df = df.set_index(str(time))
        df['idvar'] = df['alpha'].rolling(window).apply(lambda x : np.var(x*ewmaweight))
        return df
    idriskresult = idriskresult.groupby(str(stock)).apply(idrisk)
    del idriskresult[str(stock)]
    idriskresult = idriskresult.reset_index()
    idriskresult['Trddt'] = idriskresult[str(time)]
    idriskresult = idriskresult.dropna()
    '特异方差+因子方差计算'
    datelist = list(commonriskresult.keys())
    varall = dict()
    for test in datelist:
        commonrisktemp = commonriskresult[test]
        idrisktemp = idriskresult[idriskresult['Trddt'] == test]
        selectstock = list(idrisktemp[str(stock)])
        commonrisktemp = commonrisktemp[commonrisktemp[str(stock)].isin(selectstock)]
        commonrisktemp = commonrisktemp[selectstock]
        if np.allclose(np.matrix(commonrisktemp), np.matrix(commonrisktemp).T):
            totalrisk =  pd.DataFrame(np.matrix(commonrisktemp) + np.diag(idrisktemp['idvar']))
            totalrisk.columns = list(idrisktemp[str(stock)])
            totalrisk['Stkcd'] = list(idrisktemp[str(stock)])
            print(test)
            path = 'C:/Users/Han/Downloads/BarraStep/barra_stock_matrix/' + str(test)[:10]+'_barra.csv'
            totalrisk.to_csv(path)
            varall[str(test)] = totalrisk
    return varall