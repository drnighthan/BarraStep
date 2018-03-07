# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:28:09 2018

@author: Han
"""
import pandas as pd
import os
import numpy as np

monthlist = os.listdir('C:/Users/Han/Downloads/Dataset/monthly_data_2000_2017')
index = pd.DataFrame()
for i in monthlist:
    path = 'C:/Users/Han/Downloads/Dataset/monthly_data_2000_2017/'+str(i)+'/TRD_Mnth.csv'
    temp = pd.read_csv(path,parse_dates=['Trdmnt'])
    index = pd.concat([index,temp],axis=0)
value = index[['Stkcd','Trdmnt','Msmvosd']]
value.columns = ['Stkcd','Trddt','Msmvosd']
value['year'] = value['Trddt'].map(lambda x : x.year)
value['month'] = value['Trddt'].map(lambda x : x.month)
del value['Trddt']
index = index[['Stkcd','Trdmnt','Mretwd']]
index.columns = ['Stkcd','Trddt','ret+1']
index['year'] = index['Trddt'].map(lambda x : x.year)
index['month'] = index['Trddt'].map(lambda x : x.month)
del index['Trddt']
sumvalue = value.groupby(['year','month']).apply(lambda x : np.sum(x['Msmvosd']))
sumvalue = sumvalue.reset_index()
value = pd.merge(value,sumvalue,on=['year','month'])
value['Msmvosd'] = value['Msmvosd'] / value[0]
del value[0]
value = pd.merge(value,index,on=['Stkcd','year','month'])
value.to_csv('C:/Users/Han/Downloads/BarraStep/datasample/monthret.csv')