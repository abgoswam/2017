# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:06:28 2017

@author: abgoswam
"""

import pandas as pd

df_wineq_red = pd.read_csv('winequality-red.csv', sep=';')
df_wineq_white = pd.read_csv('winequality-white.csv', sep=';')

def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i+1] * row[i]
        
    return yhat
    
#b = b - learning_rate * error * x
#error = prediction - expected
#b0(t+1) = b0(t) - learning_rate * error(t)
#b1(t+1) = b1(t) - learning_rate * error(t) * x1(t)
   
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2
            coef[0] = coef[0] - (l_rate * error)
            
            for i in range(len(row) - 1):
                coef[i+1] = coef[i+1] - (l_rate * error * row[i])
                
        print("epoch={0}, lrate={1}, error={2}".format(epoch, l_rate, sum_error))    
    
    return coef 

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
l_rate = 0.001
n_epoch = 100
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)          
            
            
            
            
            
            
            