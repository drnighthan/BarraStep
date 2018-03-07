# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:29:56 2018

@author: Hanbrendan
"""
import pandas as pd
import statsmodels.api as sm 
from dateutil.parser import parse

"""因子收益率计算"""
def factorret(factor,ret,time,stock,retname,weightname):
    if type(factor) == str:
        df = pd.read_csv(factor,parse_dates=[str(time)])
        del df['Unnamed: 0']
    elif type(factor)  == pd.DataFrame:
        df = factor
        df[str(time)] = pd.to_datetime(df[str(time)])
    df['year'] = df[str(time)].map(lambda x : x.year)
    df['month'] = df[str(time)].map(lambda x : x.month)
    if type(ret) == str:
        df2 = pd.read_csv(ret,parse_dates=[str(time)])
        del df['Unnamed: 0']
    elif type(ret)  == pd.DataFrame:
        df2 = ret
        if str(time) in df2.columns:
            df2[str(time)] = pd.to_datetime(df2[str(time)])
            df2['year'] = df2[str(time)].map(lambda x : x.year)
            df2['month'] = df2[str(time)].map(lambda x : x.month)
    try:
        del df['Unnamed: 0']
        del df2['Unnamed: 0']
    except:
        pass
    df = pd.merge(df,df2,on=[str(stock),'year','month'])
    df = df.dropna()
    factorname = list(df.columns)
    outlist = ['year','month',str(time),str(stock),str(retname),str(weightname)]
    factorlist = [i for i in factorname if i not in outlist]
    def WLS_reg(df2):
        X = df2[factorlist]
        temp = sm.regression.linear_model.WLS(df2[str(retname)],X,df2[str(weightname)]).fit()
        res = pd.DataFrame(temp.params).T
        tval = pd.DataFrame(temp.tvalues).T
        tval.columns = [i+'_t' for i in factorlist]
        res = pd.concat([res,tval],axis=1)
        res['num'] = len(df)
        return res
    factor_ret = df.groupby(['year','month']).apply(WLS_reg)
    factor_ret = factor_ret.reset_index()
    factor_ret['Trddt'] = (factor_ret['year'].map(lambda x :str(x)) +'-'+ factor_ret['month'].map(lambda x :str(x)) +'-1').map(lambda x : parse(x))
    try:
        del factor_ret['level_2']
    except:
        pass
    return factor_ret


#temp= factorret('C:/Users/Hanbrendan/Downloads/barra_step/barrafactor.csv','C:/Users/Hanbrendan/Downloads/barra_step/monthret.csv','Trddt','Stkcd','ret+1','Msmvosd')