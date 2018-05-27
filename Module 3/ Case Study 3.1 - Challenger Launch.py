#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:46:50 2018

@author: timothylucas
"""

import pandas as pd
import numpy as np
from	matplotlib import pyplot as plt
import statsmodels.discrete.discrete_model as sm
from patsy import dmatrices
# Below is a fix for statsmodel to make sure the logit function works again
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

## Challenger case data

###
# Accessing the data
###

challenger_data = pd.read_csv('challenger-data.csv')

###
# Visualizing Data
###

#subsetting data
failures = challenger_data.loc[(challenger_data.Y	== 1)]
no_failures = challenger_data.loc[(challenger_data.Y == 0)]

# Frequencies

failures_freq	= failures.X.value_counts()#failures.groupby('X')
no_failures_freq =	no_failures.X.value_counts()

# Plotting

plt.scatter(failures_freq.index, failures_freq, c='red', s=40)
plt.scatter(no_failures_freq.index, np.zeros(len(no_failures_freq)),	c='blue',	s=40)
plt.xlabel('X: Temperature')
plt.ylabel('Number of Failures')
plt.show()

###
# Now for the fitting of the logistic regression
###

#get	the data	in correct format
y, X	= dmatrices('Y ~ X', challenger_data, return_type	= 'dataframe')

#build the model
logit = sm.Logit(y, X)
result = logit.fit()

# summarize the model
print(result.summary())				