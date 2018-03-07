# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:57:58 2018

@author: Han
"""
import numpy as np
import pandas as pd
from cvxopt import matrix,solvers
import os
from dateutil.parser import parse
 
def purefactor(hs300,factorloading,covlist,time,stock,factorlist):
    if type(hs300) == str:
        hs300 = pd.read_csv(hs300,parse_date=[str(time)])
    elif type(hs300) == pd.DataFrame:
        hs300[str(time)] = pd.to_datetime(hs300[str(time)])
    try:
        del hs300['Unnamed: 0']
    except:
        pass
    dirs = os.listdir(covlist)
    for z in factorlist:
        hs3001 = hs300.copy()
        hs3001[z] = hs3001[z] + 1
        print(z,hs300[z][0],hs3001[z][0])
        optim_result_all = pd.DataFrame()
        for i in dirs:
            cov = pd.read_csv((covlist+str(i)))
            if (len(cov) >0):
                del cov['Unnamed: 0']
                date = parse(i[:10])
                factorloadingtemp = factorloading[factorloading[str(time)]==date]
                hs3002 = hs3001[hs3001[str(time)]==date]
                factorloadingtemp = factorloadingtemp[factorloadingtemp[str(stock)].isin(list(cov['Stkcd']))]
                if (list(factorloadingtemp[str(stock)]) == list(cov[str(stock)]) == list(cov.columns[:-1].map(lambda x :int(x)))):
                    nrow = len(cov)
                    cov = cov.iloc[:nrow,:nrow]
                    factorloadingtempweight = factorloadingtemp.copy()
                    factorloadingtempweight = factorloadingtempweight[factorlist]
                    factorloadingtempweight['weight'] = 1
                    hs3003 = hs3002.copy()
                    hs3003 = hs3003[factorlist]
                    hs3003['weight'] =1
                    if len(hs3003) > 0:
                        P = matrix(np.matrix(cov))
                        q = matrix(np.zeros(nrow)) 
                        G = matrix(-1*np.eye(nrow))
                        h = matrix(np.zeros(nrow))
                        A = matrix(np.array(factorloadingtempweight.T))
                        b = matrix(np.array(hs3003)[0])
                        sol = solvers.qp(P,q,G,h,A,b)
                        optim_result = pd.DataFrame(np.array(sol['x']))
                        optim_result['status'] = sol['status'] 
                        optim_result['Stkcd'] = list(factorloadingtemp[str(stock)])
                        optim_result['Trddt'] = date
                        optim_result_all = pd.concat([optim_result_all,optim_result],axis=0)
                        print(optim_result.head())
        optim_result_all.to_csv('C:/Users/Han/Downloads/BarraStep/result/'+str(z)+'.csv')
