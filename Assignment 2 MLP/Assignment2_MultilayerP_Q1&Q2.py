# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:32:02 2022

@author: yiyong

Assignment 2, Q1 & Q2
"""

import numpy
import math
import matplotlib
import random

mean1 = [-2,0,1]
covariance1 = [[4,0,0],[0,5,0],[0,0,3]]

mean2 = [2,0,1]
covariance2 = [[4,0,0],[0,5,0],[0,0,3]]

data1 = numpy.random.multivariate_normal(mean1, covariance1, 1000)
data2 = numpy.random.multivariate_normal(mean2, covariance2, 1000)
Data = numpy.concatenate((data1, data2), axis=0)

Bias = numpy.full((2000,1), 1)

Data = numpy.concatenate((Bias, Data), axis=1)

class1 = numpy.full((1000,1), 0)
class2 = numpy.full((1000,1), 1)
classes = numpy.concatenate((class1, class2), axis=0)


#Randomize order of generated Data
indexArray = list(range(2000))
random.shuffle(indexArray)
Data = Data[indexArray]
classes = classes[indexArray]


n1Weight = numpy.transpose(numpy.array([(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2]))
n2Weight = numpy.transpose(numpy.array([(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2]))
n3Weight = numpy.transpose(numpy.array([(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2]))
n4Weight = numpy.transpose(numpy.array([(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2,(random.random()-.5)*2]))

#change Alpha value here
learningRate = 1
epochs = 500

TotalErrorArray = []

sameErrorCount = 0
previousError = 0

for epoch in range(epochs):
    TotalError = 0
    
    for row in range(numpy.size(Data,0)):
        
        #forward propagation
        #hidden nodes
        z1negIj = -1 * numpy.dot(Data[row,:], n1Weight)
        z1 = 1/(1 + math.exp(z1negIj))
        
        z2negIj = -1 * numpy.dot(Data[row,:], n2Weight)
        z2 = 1/(1 + math.exp(z2negIj))
        
        z3negIj = -1 * numpy.dot(Data[row,:], n3Weight)
        z3 = 1/(1 + math.exp(z3negIj))
        
        
        #output node
        hiddenInput = [1,z1,z2,z3]       
        z4NegIj = -1 * numpy.dot(hiddenInput, n4Weight)
        z4 = 1/(1 + math.exp(z4NegIj))
        
        
        #Track Errors
        guess = round(z4)
        TotalError = TotalError + abs(classes[row] - guess)
        
        
        #error propagation
        err4 = z4 * (1 - z4) * (classes[row][0] - z4)
        err1 = z1 * (1 - z1) * (err4 * n4Weight[1])
        err2 = z2 * (1 - z2) * (err4 * n4Weight[2])
        err3 = z3 * (1 - z3) * (err4 * n4Weight[3])
        
        
        #update weights
        deltaN4Weight = numpy.transpose(numpy.array([learningRate * err4 * 1, learningRate * err4 * z1, learningRate * err4 * z2, learningRate * err4 * z3]))
        n4Weight = n4Weight + deltaN4Weight
        n1Weight = n1Weight + numpy.transpose(numpy.array([learningRate * err1 * 1, learningRate * err1 * Data[row,1], learningRate * err1 *Data[row,2], learningRate * err1 *Data[row,3]]))
        n2Weight = n2Weight + numpy.transpose(numpy.array([learningRate * err2 * 1, learningRate * err2 * Data[row,1], learningRate * err2 *Data[row,2], learningRate * err2 *Data[row,3]]))
        n3Weight = n3Weight + numpy.transpose(numpy.array([learningRate * err3 * 1, learningRate * err3 * Data[row,1], learningRate * err3 *Data[row,2], learningRate * err3 *Data[row,3]]))
        
    
    print(TotalError)
    TotalErrorArray.append([TotalError[0]])
    
    if abs(TotalError - previousError) == 0:
        sameErrorCount+=1
    else:
        sameErrorCount = 0
        
    if sameErrorCount >=5:
        break

    previousError = TotalError
      
matplotlib.pyplot.plot(TotalErrorArray)

