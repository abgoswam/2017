# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:26:47 2017

@author: abgoswam
"""
import csv
import random
import pandas as pd
import numpy as np

def load_csv(filename):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        header = csv_reader.next() 
        dataset = list()
        for line in csv_reader:
            if not line:
                continue
            dataset.append(line)
    
    return dataset

def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i+1] * row[i]
        
    return yhat

# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
#        sum_error = 0
        for row in train:           
            yhat = predict(row, coef)
            error = yhat - row[-1]
#            sum_error += error**2
            coef[0] = coef[0] - l_rate * error  #b0(t+1) = b0(t) - learning_rate * error(t)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i] #b1(t+1) = b1(t) - learning_rate * error(t) * x1(t)

#        print(l_rate, n_epoch, sum_error)

    return coef

def compute_rmse(coef, test):
    print(coef)
    sum_sqdiff = 0
    for row in test:
        actual = row[-1]
        predicted = predict(row, coef)    
        sum_sqdiff += (predicted - actual) ** 2
    
    print("RMSE computed manually: {0}".format(np.sqrt((sum_sqdiff * 1.0)/len(test)) ))
    

if __name__ == "__main__":    

  #    df = pd.read_csv('winequality-red.csv', sep=';')
#    df2 = df.drop('quality', axis=1)
#    df3 = (df2 - df2.mean()) / (df2.max() - df2.min())
#    df4 = pd.concat([df3, df['quality']], axis=1)
#    df4.to_csv('winequality-red-scaled.csv', index=False)
    
    dataset = load_csv('winequality-red-scaled.csv')
    dataset = [[float(x) for x in line] for line in dataset]

    train = dataset[:1000]
    test = dataset[1000:]

    print("RMSE on test dataset  with hardcoded dummy coefficients (0)")
    coef = [0.0 for _ in range(len(test[0]))]
    compute_rmse(coef, test)

    print("RMSE on test dataset with learned coefficients")
    l_rate = 0.01
    n_epoch = 1
    coef = coefficients_sgd(train, l_rate, n_epoch)
    compute_rmse(coef, test)






