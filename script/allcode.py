# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:52:43 2018

@author: Han
"""

import matplotlib.pyplot as plt
import numpy as np 
import scipy as sc
import pandas as pd
from scipy.optimize import minimize
from dateutil.parser import parse
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hac 
from sklearn import preprocessing
import os
from pandas.tseries.offsets import MonthBegin
from dateutil.relativedelta import *
import matplotlib.pyplot as plt
import gc
import random
from cvxopt import matrix,solvers

factorlist = os.listdir('C:/Users/Han/Downloads/R code NCFR/barra/factor_database')
factorall = pd.DataFrame()
for i in factorlist:
    path = 'C:/Users/Han/Downloads/R code NCFR/barra/factor_database/' + str(i)
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
del temp['liquidityfactor'],temp['growthfactor']
temp = temp.replace([np.inf, -np.inf], np.nan)
temp = temp.dropna()
'month_select'
temp['year'] = temp['Trddt'].map(lambda x : x.year)
temp['month'] = temp['Trddt'].map(lambda x : x.month)
temp = pd.merge(temp,temp.groupby(['Stkcd','year','month'])['Trddt'].max().reset_index())
listdate = pd.read_csv('C:/Users/Han/Downloads/Dataset/listdate/TRD_Co.csv',parse_dates = ['Listdt'])
temp = pd.merge(temp,listdate,on=['Stkcd'])
temp['daydelta'] = (temp['Trddt'] > (temp['Listdt'] +pd.DateOffset(years=1)))
temp = temp[temp['daydelta']==True]
del temp['daydelta'],temp['Listdt']
'！！！date+1'
temp['Trddt'] = (temp['year'].map(lambda x:str(x)) + '-' + temp['month'].map(lambda x:str(x)) + '-1').map(lambda x:parse(x))
'factor_preprocessing'
def processing(df):
    stocklist = list(df['Stkcd'])
    trddtlist = list(df['Trddt'])
    res = df.drop(['Trddt', 'Stkcd','year','month'], axis=1)
    res = pd.DataFrame(preprocessing.scale(res))
    res.columns = ['beta', 'BP', 'earningsfactor', 'leveragefactor', 'RSTR','Non-Linear Size', 'residualvolatilityfactor', 'Size']
    print(res)
    res['Trddt'] = trddtlist
    res['Stkcd'] = stocklist
    return res 
factor_process = temp.groupby(['year','month']).apply(processing).reset_index()
del factor_process['level_2']
'returndata'
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
factor_process = pd.merge(factor_process,index,on=['Stkcd','year','month'])
sumvalue = value.groupby(['year','month']).apply(lambda x : np.sum(x['Msmvosd']))
sumvalue = sumvalue.reset_index()
value = pd.merge(value,sumvalue,on=['year','month'])
value['Msmvosd'] = value['Msmvosd'] / value[0]
del value[0]
'因子收益率计算'
ret_factorloading = pd.merge(factor_process,value,on=['Stkcd','year','month'])
def factorret(df):
    df = df.dropna()
    X = df[['beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size']]
    temp = sm.regression.linear_model.WLS(df['ret+1'],X,df['Msmvosd']).fit()
    res = pd.DataFrame(temp.params).T
    tval = pd.DataFrame(temp.tvalues).T
    tval.columns = ['beta_t', 'BP_t', 'earningsfactor_t', 'leveragefactor_t','RSTR_t', 'Non-Linear Size_t', 'residualvolatilityfactor_t', 'Size_t']
    res = pd.concat([res,tval],axis=1)
    res['num'] = len(df)
    return res
factor_ret = ret_factorloading.groupby(['year','month']).apply(factorret)
factor_ret = factor_ret.reset_index()
del factor_ret['level_2']
'方差矩阵计算'
def weightcreate(x,num):
    weight = (0.5)**(x)
    weightlist = [(weight**(num-i))**2 for i in range(num)]
    return weightlist
ewmaweight = weightcreate(1/63,12)
factor_ret['Trddt'] = (factor_ret['year'].map(lambda x :str(x)) +'-'+ factor_ret['month'].map(lambda x :str(x)) +'-1').map(lambda x : parse(x))
factorcov = factor_ret[['Trddt','beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size']] 
factorcov = factorcov.sort_values('Trddt')
def cov_matrix(df,window):
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
factorcov = cov_matrix(factorcov,12)
'B调整'
datelist = list(factorcov['Trddt'].drop_duplicates())
b2 = pd.DataFrame()
for i in datelist:
    b2temp = pd.DataFrame()
    temp = factorcov[factorcov['Trddt'] == i]
    temp = temp.set_index(['Trddt'])
    temp2 = pd.DataFrame(np.diag(temp)).T
    temp2.columns = temp.columns.map(lambda x : str(x)+'_cov')
    tempret = pd.DataFrame(factor_ret[factor_ret['Trddt']==i][list(temp.columns)]).reset_index(drop=True)
    temp3 = pd.concat([temp2,tempret],axis=1)
    b2temp['B2'] = (1/len(temp.columns) * sum([temp3[i]**2/temp3[str(i)+'_cov'] for i in list(temp.columns)]))**0.5
    b2temp['Trddt'] = i
    b2 = pd.concat([b2,b2temp],axis=0)
ewmaweight = weightcreate(1/23,12)
b2['lambda'] = b2['B2'].rolling(12).apply(lambda x : np.dot(x**2,ewmaweight)**0.5)
factor_ret = pd.merge(factor_ret,b2,on=['Trddt'])
factor_ret['B'] = factor_ret['B2'].map(lambda x:x**0.5)
factor_ret = factor_ret.dropna()
factor_ret.loc[factor_ret['B'] <1, 'lambda'] = 1
factorcov_lambda = pd.DataFrame()
datelist = list(factor_ret['Trddt'].drop_duplicates())
for i in datelist:
    lambda2 = factor_ret[factor_ret['Trddt']==i]['lambda'].astype(float)**2
    temp = factorcov[factorcov['Trddt']==i]
    del temp['Trddt']
    temp = temp.multiply(np.float(lambda2))
    temp['Trddt'] =i
    factorcov_lambda = pd.concat([factorcov_lambda,temp],axis=0)
#statsmodels.stats.sandwich_covariance.cov_hac

'NW调整'
#num = list(factorcov_lambda['Trddt'].drop_duplicates().sort_values())
#Zdict = {}
#for t in range(len(num)):
#    temp =  factorcov_lambda[factorcov_lambda['Trddt'] == num[t]]
#    del temp['Trddt']
#    Zdict[t]  = temp
#Zsum = {}
#datelist = list(factorcov_lambda['Trddt'].drop_duplicates().sort_values())
#for t in range(11,len(datelist)):
#    Zdictsum = dict([(k, Zdict[k]) for k in Zdict if (k <= t) and (k >= (t-11))]) 
#    Zsum[t] = sum(Zdictsum.values())/len(Zdictsum.keys())
#def newey_west_adj_cov(df,T,m):
#    '要改成矩阵形式'
#    datelist = list(df['Trddt'].drop_duplicates().sort_values())
#    result_df = pd.DataFrame()
#    temp2 = {}
#    for t in range(11,len(datelist)):
#        cal = {}
#        Z_bar_t= Zsum[t]
#        tempdatelist = datelist[(t-11):(t+1)]
#        for z in range(0,12):
#            temp = factorcov_lambda[factorcov_lambda['Trddt'] == tempdatelist[z]]
#            del temp['Trddt']
#            cal[z] = temp - Z_bar_t
#        def r_series(j):
#            r_0 = 0
#            for z in range(j):
#                cal1 = cal[z]
#                cal2 = cal[(len(cal.keys())-1-z)]
#                r_0 += cal1*cal2
#            return(r_0)
#        r_result = dict([(j, r_series(j)* 1/12) for j in range(12)]) 
#        r_0 = r_result[0]
#        m = int(4*(12/100)**(2/9))
#        w_series = np.array([(1- j/(m+1)) for j in range(1,(m+1))])
#        tempsum = pd.DataFrame()
#        for i in range(1,m):
#            temp = r_result[i]
#            temp = temp * w_series[(i-1)]
#            if len(tempsum) ==0:
#                tempsum = temp
#            else:
#                tempsum += temp
#        F = r_0+2*tempsum
#        print(F)
#        temp2[datelist[t]] = F
#    return result_df
'因子方差计算'
factor_ret = factor_ret[['Trddt','beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size']]
ret_factorloading['Trddt'] = (ret_factorloading['year'].map(lambda x :str(x)) +'-'+ ret_factorloading['month'].map(lambda x :str(x)) +'-1').map(lambda x : parse(x))
factorloading = ret_factorloading[['Stkcd','Trddt','beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size']]
datelist = list(factorcov_lambda['Trddt'].drop_duplicates())
commonriskresult = dict()
for i in datelist:
    tempfactorloading = factorloading[factorloading['Trddt'] == i]
    tempfactorcov = factorcov_lambda[factorcov_lambda['Trddt'] == i]
    del tempfactorcov['Trddt']
    faloadtemp = tempfactorloading[tempfactorcov.columns]
    tempres = pd.DataFrame(np.dot(np.dot(faloadtemp,tempfactorcov),faloadtemp.T))
    tempres.columns = list(tempfactorloading['Stkcd'])
    tempres['Stkcd'] = list(tempfactorloading['Stkcd'])
    commonriskresult[i] = tempres
del temp,tempfactorcov,tempfactorloading,index,sumvalue
'特异方差计算'
idriskresult = pd.merge(ret_factorloading,factor_ret,on='Trddt')
factorlist = ['beta','BP','earningsfactor','leveragefactor','RSTR','Non-Linear Size','residualvolatilityfactor','Size']
temp = pd.DataFrame([idriskresult[str(i)+'_x'] * idriskresult[str(i)+'_y'] for i in factorlist]).T
temp = temp.sum(axis=1)
idriskresult['alpha'] = idriskresult['ret+1'] - temp
idriskresult = idriskresult[['Stkcd','Trddt','alpha']]
def idrisk(df):
    df = df.sort_values('Trddt')
    df = df.set_index('Trddt')
    df['idvar'] = df['alpha'].rolling(12).apply(lambda x : np.var(x*ewmaweight))
    return df
idriskresult = idriskresult.groupby('Stkcd').apply(idrisk)
del idriskresult['Stkcd']
idriskresult = idriskresult.reset_index()
idriskresult['Trddt'] = idriskresult['Trddt']
idriskresult = idriskresult.dropna()
'特异方差+因子方差计算'
datelist = list(commonriskresult.keys())
for test in datelist:
    commonrisktemp = commonriskresult[test]
    idrisktemp = idriskresult[idriskresult['Trddt'] == test]
    selectstock = list(idrisktemp['Stkcd'])
    commonrisktemp = commonrisktemp[commonrisktemp['Stkcd'].isin(selectstock)]
    commonrisktemp = commonrisktemp[selectstock]
    print(np.allclose(np.matrix(commonrisktemp), np.matrix(commonrisktemp).T))
    totalrisk =  pd.DataFrame(np.matrix(commonrisktemp) + np.diag(idrisktemp['idvar']))
    totalrisk.columns = list(idrisktemp['Stkcd'])
    totalrisk['Stkcd'] = list(idrisktemp['Stkcd'])
    print(test)
    path = 'C:/Users/Han/Downloads/R code NCFR/barra/barra_stock_matrix_barra_factors/' + str(test)[:10]+'_barra.csv'
    totalrisk.to_csv(path)
'hs300_factoringloading'
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
def purefactor(factorloading):
    dirs = os.listdir('C:/Users/Han/Downloads/R code NCFR/barra/barra_stock_matrix_barra_factors/')
    for z in list(hs300.columns)[1:]:
        hs3001 = hs300.copy()
        hs3001[z] = hs3001[z] + 1
        print(z,hs300[z][0],hs3001[z][0])
        optim_result_all = pd.DataFrame()
        for i in dirs:
            cov = pd.read_csv(('C:/Users/Han/Downloads/R code NCFR/barra/barra_stock_matrix_barra_factors/'+str(i)))
            if (len(cov) >0):
                del cov['Unnamed: 0']
                date = parse(i[:10])
                factorloadingtemp = factorloading[factorloading['Trddt']==date]
                hs3002 = hs3001[hs3001['Trddt']==date]
                factorloadingtemp = factorloadingtemp[factorloadingtemp['Stkcd'].isin(list(cov['Stkcd']))]
                if (list(factorloadingtemp['Stkcd']) == list(cov['Stkcd']) == list(cov.columns[:-1].map(lambda x :int(x)))):
                    nrow = len(cov)
                    cov = cov.iloc[:nrow,:nrow]
                    factorloadingtempweight = factorloadingtemp.copy()
                    factorloadingtempweight['weight'] = 1
                    hs3003 = hs3002.copy()
                    hs3003['weight'] =1
                    P = matrix(np.matrix(cov))
                    q = matrix(np.zeros(nrow)) 
                    G = matrix(-1*np.eye(nrow))
                    h = matrix(np.zeros(nrow))
                    A = matrix(np.array(factorloadingtempweight.iloc[:,2:].T))
                    b = matrix(np.array(hs3003.iloc[:,1:])[0])
                    sol = solvers.qp(P,q,G,h,A,b)
                    optim_result = pd.DataFrame(np.array(sol['x']))
                    optim_result['status'] = sol['status'] 
                    optim_result['Stkcd'] = list(factorloadingtemp['Stkcd'])
                    optim_result['Trddt'] = date
                    optim_result_all = pd.concat([optim_result_all,optim_result],axis=0)
                    print(optim_result.head())
        optim_result_all.to_csv('C:/Users/Han/Downloads/R code NCFR/barra/purefactortoresult - T-1_barra_factors/'+str(z)+'.csv')
purefactor(factorloading)
'backtest'
data = pd.read_csv('C:/Users/Han/Downloads/R code NCFR/barra/purefactortoresult - T-1_barra_factors/BP.csv',parse_dates = ['Trddt'])
monthlist = os.listdir('C:/Users/Han/Downloads/Dataset/monthly_data_2000_2017')
index = pd.DataFrame()
for i in monthlist:
    path = 'C:/Users/Han/Downloads/Dataset/monthly_data_2000_2017/'+str(i)+'/TRD_Mnth.csv'
    temp = pd.read_csv(path,parse_dates=['Trdmnt'])
    index = pd.concat([index,temp],axis=0)
index = index[['Stkcd','Trdmnt','Mretnd']]
index.columns = ['Stkcd','Trddt','Mretnd']
data = pd.merge(data,index,on=['Stkcd','Trddt'])
data['retper'] = data['0'] * data['Mretnd']
BP= data.groupby(['Trddt']).apply(lambda x:sum(x['retper']))


