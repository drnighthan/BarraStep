# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:42:07 2018

@author: Han
"""

import pandas as pd
import factor_standardize
import factor_ret_wls
import factor_ewma_cov
import covadjustB
import stockcommonrisk
import optimizer
barrafactor = pd.read_csv('C:/Users/Han/Downloads/BarraStep/datasample/barrafactor.csv')
barrafactor = factor_standardize.processing(barrafactor,'Trddt','Stkcd')
returndata = pd.read_csv('C:/Users/Han/Downloads/BarraStep/datasample/monthret.csv')
factorret = factor_ret_wls.factorret(barrafactor,returndata,"Trddt","Stkcd","ret+1","Msmvosd")
factorcov = factor_ewma_cov.ewma_cov_matirx(factorret,12,'Trddt',['beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size'])
factorcov_lambda = covadjustB.B_adjustment(factorcov,factorret,'Trddt',12)
returndata['Trddt'] = pd.to_datetime(returndata['year'].map(lambda x:str(x)) + '-' + returndata['month'].map(lambda x:str(x)) + '-1')
returndata = returndata[['Stkcd','Trddt','ret+1']]
barrafactor = pd.merge(barrafactor,returndata,on=['Trddt','Stkcd'])
covdict = stockcommonrisk.risk(factorret,barrafactor,['beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size'],factorcov_lambda,'Trddt','Stkcd','ret+1',12)
hs300sample = pd.read_csv('C:/Users/Han/Downloads/BarraStep/datasample/sample_hs300.csv')
optimizer.purefactor(hs300sample,barrafactor,'C:/Users/Han/Downloads/BarraStep/barra_stock_matrix/',"Trddt","Stkcd",['beta', 'BP', 'earningsfactor', 'leveragefactor','RSTR', 'Non-Linear Size', 'residualvolatilityfactor', 'Size'])
