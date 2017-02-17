# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:26:47 2017

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

def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i+1] * row[i]
        
    return yhat

    
filename = 'winequality-red.csv'
dataset = load_csv(filename)
dataset = [[float(x) for x in line] for line in dataset]

train = dataset[:1000]
test = dataset[1000:]


coef = [0.0 for _ in range(len(test[0]))]
sum_sqdiff = 0
for row in test:
    actual = row[-1]
    predicted = predict(row, coef)
    print(actual, predicted)
    
    sum_sqdiff += (predicted - actual) ** 2
    
print("MSE computed manually: {0}".format((sum_sqdiff * 1.0)/len(test)))