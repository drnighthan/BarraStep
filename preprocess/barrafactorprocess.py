# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:30:43 2018

@author: Hanbrendan
"""

import pandas as pd
import os
import numpy as np
from dateutil.parser import parse
factorlist = os.listdir('C:/Users/Hanbrendan/Downloads/R code NCFR/barra/factor_database')
factorall = pd.DataFrame()
for i in factorlist:
    path = 'C:/Users/Hanbrendan/Downloads/R code NCFR/barra/factor_database/' + str(i)
    x = pd.HDFStore(path)
    temp = x['data']
    try:
        del temp['level_0']
    except:
        if len(factorall) == 0:
            factorall = temp.copy()
        else:
            factorall = pd.merge(factorall,temp,how='outer',on=['Stkcd','Trddt'])
    print(factorall)
temp = factorall.sort_values(['Stkcd','Trddt'])
temp = temp.groupby('Stkcd').apply(lambda x : x.fillna(method='ffill'))
'数据太少，将liquidity和Growth去掉'
del temp['liquidityfactor'],temp['growthfactor']
temp = temp.replace([np.inf, -np.inf], np.nan)
temp = temp.dropna()
'month_select'
temp['year'] = temp['Trddt'].map(lambda x : x.year)
temp['month'] = temp['Trddt'].map(lambda x : x.month)
temp = pd.merge(temp,temp.groupby(['Stkcd','year','month'])['Trddt'].max().reset_index())
listdate = pd.read_csv('C:/Users/Hanbrendan/Downloads/Dataset/listdate/TRD_Co.csv',parse_dates = ['Listdt'])
temp = pd.merge(temp,listdate,on=['Stkcd'])
temp['daydelta'] = (temp['Trddt'] > (temp['Listdt'] +pd.DateOffset(years=1)))
temp = temp[temp['daydelta']==True]
del temp['daydelta'],temp['Listdt']
'！！！date+1'
temp['Trddt'] = temp['Trddt'] + pd.DateOffset(months=1)
temp['year'] = temp['Trddt'].map(lambda x : x.year)
temp['month'] = temp['Trddt'].map(lambda x : x.month)
temp['Trddt'] = (temp['year'].map(lambda x:str(x)) + '-' + temp['month'].map(lambda x:str(x)) + '-1').map(lambda x:parse(x))
del temp['month'],temp['year']
temp.to_csv('C:/Users/Hanbrendan/Downloads/BarraStep/barrafactor.csv')