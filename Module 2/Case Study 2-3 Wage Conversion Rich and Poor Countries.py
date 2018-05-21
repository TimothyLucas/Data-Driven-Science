#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:05:40 2018

@author: timothylucas
"""

import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm

def bootstrap_conf_interval():
    # This function was written to make sure the bootstrap method to
    # obtain the confidence interval could be obtained, which is not 
    # provided by sklearn by default
    # Theory on bootstrap: https://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
    # Implementation in Python (with leastsquares, see highest voted answer): https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
#    return std_error
    return None

###
# Load data
###

growth = pd.read_csv('growth_figures.csv')
growth = growth.set_index('Unnamed: 0')

# define formulas

col_names = growth.columns

xnames = col_names[3:]
dandxnames = col_names[2:]

fmla = 'Outcome ~ '+'+'.join(dandxnames)
fmla_y = 'Outcome ~ '+'+'.join(xnames)
fmla_d = 'gdpsh465 ~ '+'+'.join(xnames)

# make matrices

y, X = dmatrices(fmla, growth)
y_y, X_y = dmatrices(fmla_y, growth)
y_d, X_d = dmatrices(fmla_d, growth)

# Fit initial model, should not be so good, as witnessed by the confidence 
# intevals and P values of the summary

X2 = sm.add_constant(X)
model = sm.OLS(y, X2)
est_ols = model.fit()

# Original estimate, we can see that the coeff for gdpsh465 is the same
# as in R

est_ols.summary()

# Now let's apply lasso to the d and y models, and after that we to take 
# the residuals, do a 'normal' OLS on this and then we have partialled them out
# the only problem here is the setting of the alpha, which I am not sure of 
# which should be the correct value, since the R documenatation needs 
# two penalty numbers, and the stats model only one, so not sure
# how to fix this... (bit more research would work probably, but
# I've already spent enough time on this)

X2_y = sm.add_constant(X_y)
model_y = sm.OLS(y_y, X2_y)
est_y_lasso = model_y.fit_regularized(L1_wt=1, alpha=1)
res_y = np.concatenate(y_y) - est_y_lasso.fittedvalues

X2_d = sm.add_constant(X_d)
model_d = sm.OLS(y_d, X2_d)
est_d_lasso = model_d.fit_regularized(L1_wt=1, alpha=1)
res_d = np.concatenate(y_d) - est_d_lasso.fittedvalues

# Now for the partialling out

df_partial_out = pd.DataFrame(columns = ['res_y', 'res_d'],\
                              data = np.transpose([res_y, res_d]))
partial_y, partial_X =  dmatrices('res_y ~ res_d - 1', df_partial_out)

X2_partial = sm.add_constant(partial_X)
model_partial = sm.OLS(partial_y, X2_partial)
est_partial_ols = model_partial.fit()

est_partial_ols.summary()

# So this doesn't seem to be working particularly well, not sure where this goes
# wrong...
