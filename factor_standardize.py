# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:47:40 2018

@author: Hanbrendan
"""
import pandas as pd
from sklearn import preprocessing

"""因子的预处理，这里主要是对因子的标准化"""
def processing(df,time,stock):
    if type(df) == str:
        df = pd.read_csv(df,parse_dates=[str(time)])
        del df['Unnamed: 0']
    elif type(df)  == pd.DataFrame:
        df[str(time)] = pd.to_datetime(df[str(time)])
    df['year'] = df[str(time)].map(lambda x : x.year)
    df['month'] = df[str(time)].map(lambda x : x.month)
    def stand(df2):
        stocklist = list(df2[str(stock)])
        trddtlist = list(df2[str(time)])
        res = df2.drop([str(time), str(stock),'year','month'], axis=1)
        namelist = list(res.columns)
        res = pd.DataFrame(preprocessing.scale(res))
        res.columns = namelist
        res['Trddt'] = trddtlist
        res['Stkcd'] = stocklist
        return res 
    factor_process = df.groupby(['year','month']).apply(stand).reset_index()
    try:
        del factor_process['level_2']
    except:
        pass
    return factor_process
