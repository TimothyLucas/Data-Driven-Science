#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 20:54:33 2018

@author: timothylucas
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm


def r2_adj(R_squared, obs, indep_vars):
    return 1 - (1- R_squared)*((obs-1)/(obs-indep_vars-1))

def summarizeResults(y, X, lm_fitted):
    # Lazy function to not have to copy all of the lines all of the time,
    # and for consistency off course, also very important...
    
    # First for the R_squared
    preds = lm_fitted.predict(X)
    R2 = r2_score(y, preds)
    
    # Then for the R_squared_adjusted
    
    R2_adj = r2_adj(R2, X.shape[0], X.shape[1])
    
    # lastly for the MSE
    
    MSE = mean_squared_error(y, preds)
    
    return R2, R2_adj, MSE

# In this file I have translated the R file into Python 
# Since the calculations are pretty straightforward 
# I have not put everything in a function, but it's just a 
# simple script

###
# Load data
###

data = pd.read_csv('wage_gap_data.csv')
data = data.set_index('Unnamed: 0')

###
# Define models
###

# Number of variables in not the same though in Python and R even though
# same formulas are being used, is this the source for the error?


y_basic, X_basic = dmatrices('wage ~ female + sc+ cg+ mw + so + we + exp1 + exp2 + exp3', data)
y_flex, X_flex = dmatrices('wage ~ female + (sc+ cg+ mw + so + we + exp1 + exp2 + exp3)**2', data)

###
# Run Linear Regression
###

lm_basic = LinearRegression()
lm_flex = LinearRegression()

basic_fitted = lm_basic.fit(X_basic, y_basic)
flex_fitted = lm_flex.fit(X_flex, y_flex)


###
# Print and summarize results
###

# R squared, R squared adjusted (not same as in R)
# and MSE (also not same as in R calculations for some reason)

basic_r2, basic_r2_adj, basic_MSE = summarizeResults(y_basic, X_basic, basic_fitted)
flex_r2, flex_r2_adj, flex_MSE = summarizeResults(y_flex, X_flex, flex_fitted)

###
# Train/Test split
###

train_X_basic, test_X_basic, train_y_basic, test_y_basic = train_test_split(X_basic, y_basic, \
                                                    test_size=0.5)

train_X_flex, test_X_flex, train_y_flex, test_y_flex = train_test_split(X_flex, y_flex, \
                                                    test_size=0.5)

###
# Do calculations again
###

basic_r2_s, basic_r2_adj_s, basic_MSE_s = summarizeResults(train_y_basic, train_X_basic, basic_fitted)
flex_r2_s, flex_r2_adj_s, flex_MSE_s = summarizeResults(train_y_flex, train_X_flex, flex_fitted)

###
# Summarize results in a table
###

results = pd.DataFrame([[X_basic.shape[1], basic_r2, basic_r2_adj, basic_MSE],\
                        [X_flex.shape[1], flex_r2, flex_r2_adj, flex_MSE], \
                        [train_X_basic.shape[1], basic_r2_s, basic_r2_adj_s, basic_MSE_s], \
                        [train_X_flex.shape[1], flex_r2_s, flex_r2_adj_s, flex_MSE_s]], \
                       columns = ['p', 'R2', 'R2 Adjusted', 'MSE'], \
                       index = ['Basic Model', 'Flex Model', \
                                'Basic Model (50% Train)', \
                                'Flex Model (50% Train)'])

### 
# Continuing with Case Study 2.2 (since data used and models are the same)
###

# So first calculate the stats

stats_female = data.loc[data['female'] == 1].apply(np.mean, axis = 0)
stats_male = data.loc[data['female'] == 0].apply(np.mean, axis = 0)

sum_stats = pd.concat([stats_female, stats_male], axis = 1)
sum_stats.columns = ['female_stats', 'male_stats']

###
# Calculation of other relevant statistics
###

# Now perform the linear regression (not again, since we've already done
# that above), and calculate the Estimate, the Standard error, and the 
# confidence bounds for the regression
# I've used the statsmodel package here since it provides easier access to 
# p values and the like (and also through all of the other statistics we
# were looking for before)

X2_basic = sm.add_constant(X_basic)
est_basic = sm.OLS(y_basic, X2_basic)
est2_basic = est_basic.fit()

# Note, statsmodel provides the parameters below for all model independent 
# variables, we're just selecting out the ones relevant for 'female'

coef_basic = est2_basic.params[1]
stderr_basic = est2_basic.bse[1]
conf_basic = est2_basic.conf_int()[1]

params_basic = np.append([coef_basic, stderr_basic], conf_basic)

# Now for the flex model

X2_flex = sm.add_constant(X_flex)
est_flex = sm.OLS(y_flex, X2_flex)
est2_flex = est_flex.fit()

coef_flex = est2_flex.params[1]
stderr_flex = est2_flex.bse[1]
conf_flex = est2_flex.conf_int()[1]

params_flex = np.append([coef_flex, stderr_flex], conf_flex)

# Put together and print

results_wg = pd.DataFrame([params_basic, params_flex], \
                          columns = ['Coeff', 'Std Error', 'Lower Bound (95% int)', 'Upper Bound (95% int)'], \
                          index = ['Basic Model', 'Flex Model'])
print(results_wg)

# As can be seen, the same results are obtained here as compared to the R script

###
# Partialling out
###

# So let's now do the partialling out

# First the formulas

fmla2_y = 'wage ~  (sc+ cg+ mw + so + we + exp1 + exp2 + exp3)**2'
fmla2_d = 'female ~ (sc+ cg+ mw + so + we + exp1 + exp2 + exp3)**2'

# Generate the matrices

y_y, X_y = dmatrices(fmla2_y, data)
y_d, X_d = dmatrices(fmla2_d, data)

# Fit and obtain the residuals

X2_y = sm.add_constant(X_y)
model_y = sm.OLS(y_y, X2_y)
est_y = model_y.fit()
res_y = np.concatenate(y_y) - est_y.fittedvalues

X2_d = sm.add_constant(X_d)
model_d = sm.OLS(y_d, X2_d)
est_d = model_d.fit()
res_d = np.concatenate(y_d) - est_d.fittedvalues

# Now for the partialling out, results are the same as with R, so that
# is great!

df_partial_out = pd.DataFrame(columns = ['res_y', 'res_d'],\
                              data = np.transpose([res_y, res_d]))
partial_y, partial_X =  dmatrices('res_y ~ res_d', df_partial_out)

X2_partial = sm.add_constant(partial_X)
model_partial = sm.OLS(partial_y, X2_partial)
est_partial_ols = model_partial.fit()

est_partial_ols.summary()

###
# Other notes
###

# If you look at the summary from the linear regression you can also see that the 
# biggest boost to wage is (still) a college degree, so hope that one is paying
# off for you :)

# Would also be interesting to see how the wages differ per region, and
# so on, but that is for another time...


