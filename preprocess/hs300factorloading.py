# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:05:38 2018

@author: Han
"""
import os 
import pandas as pd
from dateutil.parser import parse
factorloading = pd.read_csv('C:/Users/Han/Downloads/barra_step/sample_factorloading.csv',parse_dates=['Trddt'])
dirs = os.listdir('C:/Users/Han/Downloads/Dataset/index_percent_2005_2016/')
hs300 = pd.DataFrame()
for i in dirs:
    path = 'C:/Users/Han/Downloads/Dataset/index_percent_2005_2016/'+str(i)+'/IDX_Smprat.csv'
    temp = pd.read_csv(path,encoding = 'GBK',parse_dates = ['Enddt'])
    hs300 = pd.concat([hs300,temp])
hs300 = hs300[hs300['Indexcd']==300]
hs300['year'] = hs300['Enddt'].map(lambda x : x.year)
hs300['month'] = hs300['Enddt'].map(lambda x : x.month)
temp = hs300.groupby(['year','month']).apply(lambda x : max(x['Enddt'])).reset_index()
temp = temp.rename(columns= {0:'Enddt'})
hs300 = pd.merge(hs300,temp,on=['year','month','Enddt'])
hs300['Trddt'] = (hs300['year'].map(lambda x : str(x)) +'-'+hs300['month'].map(lambda x : str(x))+'-1').map(lambda x : parse(x))
hs300 = hs300[['Stkcd','Weight','Trddt']]
hs300 = pd.merge(hs300,factorloading,on=['Stkcd','Trddt'])
sumlist = list(hs300.columns)[3:]
for i in sumlist:
    hs300[i] = hs300[i]*hs300['Weight']/100
hs300 = hs300.groupby('Trddt').apply(sum)
hs300 = hs300.reset_index()
del hs300['Stkcd'],hs300['Weight']
hs300.to_csv('C:/Users/Han/Downloads/barra_step/sample_hs300.csv')