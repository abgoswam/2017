# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:09:17 2017

@author: abgoswam
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('winequality-red.csv', sep=';')

train = df[:1000]
test = df[1000:]

train_X, train_Y = train.drop(['quality'], axis=1), train['quality']
test_X, test_Y = test.drop(['quality'], axis=1), test['quality']

#Train on train data
lr = LinearRegression()
lr.fit(train_X, train_Y)

print lr
print 'coefficient: ',lr.coef_
print 'intercept: ',lr.intercept_

#Test on test data
df_Predictions = lr.predict(test_X)

# Compute MSE using sklearn
print("MSE using sklearn: {0}".format(mean_squared_error(test_Y, df_Predictions)))

#Compute MSE manually
sum_sqdiff = 0
for actual, predicted in zip(test_Y, df_Predictions):
    sum_sqdiff += (predicted - actual) ** 2
    
print("MSE computed manually: {0}".format((sum_sqdiff * 1.0)/len(test_Y)))

#Alternate way
err = abs(test_Y - df_Predictions)
total_error = np.dot(err,err)
print("MSE computed alternate: {0}".format((total_error * 1.0)/len(test_Y)))