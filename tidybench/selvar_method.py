"""
Implements the SELVAR (Selective auto-regressive model) algorithm.

Based on an implementation that is originally due to Gherardo Varando (gherardovarando).
"""


import numpy as np
###
# compile selvar.f with: 
#    f2py -llapack -c -m selvar selvar.f
###
from selvar import slvar, gtstat, gtcoef
from scipy.stats import chi2

stats = ["DF", "LR", "FS"] 
coefs = ["ABS", "SQR", "COEF"] 
def interpnans(y):
    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def my_method(data,
           maxlags = 1,
           batchsize = -1,
           score = "default",
           nrm = 1,
           mxitr = -1,
           trace = 0):
    
     # replace missing values coded as 999
     data[data == 999] = np.nan
     for j in range(data.shape[1]):
        data[:, j] = interpnans(data[:, j])

     ml = int(maxlags)
     
     scores, a, info = slvar(data, bs = batchsize, ml = ml, mxitr = mxitr, trc = trace)      
     stat, df = gtstat(data, a = a, bs = batchsize, ml = ml, job = "LR") 
     
     pvalues = 1 - chi2.cdf(stat, df[:,0] - df[:,1])  
     pvalues = pvalues * data.shape[1] 
     pvalues[pvalues > 1] = 1
     lags = a 
     
     if score in  stats:
        scores, df = gtstat(data, a = a, bs = batchsize, ml = ml, job = score) 
     if score in coefs:
        scores, info = gtcoef(data, a = a, bs = batchsize, ml = ml, job = score, nrm = nrm)
     scores = abs(scores)

     return scores, pvalues, lags
