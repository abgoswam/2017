# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:09:17 2017

@author: abgoswam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('winequality-red.csv', sep=';')

train = df[:1000]
test = df[1000:]

train_X, train_Y = train.drop(['quality'], axis=1), train['quality']
test_X, test_Y = test.drop(['quality'], axis=1), test['quality']

#----------------------------------

#Train on train data
lr = LinearRegression()
lr.fit(train_X, train_Y)

print(lr)
print('coefficient: {0}').format(lr.coef_)
print('intercept: {0}').format(lr.intercept_)

#Test on test data
test_Predictions = lr.predict(test_X)

plt.scatter(np.array(test_Y), test_Predictions)
plt.ylabel('predicted')
plt.xlabel('real')
plt.show()

# Compute MSE using sklearn
#print("GOF: {0}".format(1 - (mean_squared_error(test_Y, test_Predictions) / np.var(test_Y))))
print("RMSE using sklearn: {0}".format(np.sqrt(mean_squared_error(test_Y, test_Predictions))))

#Compute MSE manually
sum_sqdiff = 0
for actual, predicted in zip(test_Y, test_Predictions):
    sum_sqdiff += (predicted - actual) ** 2
    
print("MSE computed manually: {0}".format(np.sqrt((sum_sqdiff * 1.0)/len(test_Y))))

#Alternate way
err = abs(test_Y - test_Predictions)
total_error = np.dot(err,err)
print("MSE computed alternate: {0}".format(np.sqrt((total_error * 1.0)/len(test_Y))))

#----------------------------------------

scaler = StandardScaler()
scaler.fit(train_X)  # Don't cheat - fit only on training data
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)  # apply same transformation to test data

clf = linear_model.SGDRegressor(n_iter=1)
clf.fit(train_X, train_Y)
print clf
print('coefficient: {0}').format(clf.coef_)
print('intercept: {0}').format(clf.intercept_)

#Test on test data
test_Predictions = clf.predict(test_X)

plt.scatter(np.array(test_Y), test_Predictions)
plt.ylabel('predicted')
plt.xlabel('real')
plt.show()

# Compute MSE using sklearn
print("GOF: {0}".format(1 - (mean_squared_error(test_Y, test_Predictions) / np.var(test_Y))))
print("RMSE using sklearn: {0}".format(np.sqrt(mean_squared_error(test_Y, test_Predictions))))

#Compute MSE manually
sum_sqdiff = 0
for actual, predicted in zip(test_Y, test_Predictions):
    sum_sqdiff += (predicted - actual) ** 2
    
print("RMSE computed manually: {0}".format(np.sqrt((sum_sqdiff * 1.0)/len(test_Y))))

#Alternate way
err = abs(test_Y - test_Predictions)
total_error = np.dot(err,err)
print("RMSE computed alternate: {0}".format(np.sqrt((total_error * 1.0)/len(test_Y))))

# --------------------




