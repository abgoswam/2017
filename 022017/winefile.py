# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:24:00 2017

@author: abgoswam
"""

import csv
import random

def load_csv(filename):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f, delimiter=';')
        header = csv_reader.next() 
        dataset = list()
        for line in csv_reader:
            if not line:
                continue
            dataset.append(line)
    
    return dataset

def dataset_minmax(dataset):
    minmax = list()    
    for i in range(len(dataset[0])):
        ithcolvalues = [row[i] for row in dataset]
        ithcol_min = min(ithcolvalues)
        ithcol_max = max(ithcolvalues)
        minmax.append([ithcol_min, ithcol_max])
        
    return minmax
        
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds) 
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
  
	return dataset_split

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat
 
# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]

        print(l_rate, n_epoch, sum_error)

    return coef
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(predictions)

        
if __name__ == "__main__":
    filename = 'winequality-red.csv'
    dataset = load_csv(filename)
    dataset = [[float(x) for x in line] for line in dataset]
        
    # normalize
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    
    folds = cross_validation_split(dataset, 3)
    
    train = folds[0] + folds[1]
    test = folds[2]
    l_rate = 0.001
    n_epoch = 100
    predictions = linear_regression_sgd(train, test, l_rate, n_epoch)