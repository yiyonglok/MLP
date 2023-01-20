# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:17:20 2022

@author: yiyong

Assignment 2 Q3
"""

import numpy
import matplotlib
import pandas
import random
import warnings

warnings.filterwarnings('ignore')

pandaData = pandas.read_excel('spam.xlsx')
#print(pandaData)
Data = pandas.DataFrame.to_numpy(pandaData)

#Randomize order of generated Data
indexArray = list(range(numpy.size(Data,0)))
random.shuffle(indexArray)
Data = Data[indexArray]

#Separate Classes Column from Data
classes = Data[:, -1]
Data = Data[:, :-1]

#print(Data)
#print(classes)

#Add the bias input to start of data set
Bias = numpy.full((numpy.size(Data,0), 1), 1)
Data = numpy.concatenate((Bias, Data), axis=1)

#hidden layer neuron weights
n1Weight = []
n2Weight = []
n3Weight = []
n4Weight = []
n5Weight = []
n6Weight = []

for column in range(numpy.size(Data, 1)):
    n1Weight.append((random.random()-.5)*2)
    n2Weight.append((random.random()-.5)*2)
    n3Weight.append((random.random()-.5)*2)
    n4Weight.append((random.random()-.5)*2)
    n5Weight.append((random.random()-.5)*2)

n1Weight = numpy.transpose(numpy.array(n1Weight))
n2Weight = numpy.transpose(numpy.array(n2Weight))
n3Weight = numpy.transpose(numpy.array(n3Weight))
n4Weight = numpy.transpose(numpy.array(n4Weight))
n5Weight = numpy.transpose(numpy.array(n5Weight))

#output neuron weight
n6Weight = numpy.transpose(numpy.array([(random.random()-.5)*2,
                                        (random.random()-.5)*2,
                                        (random.random()-.5)*2,
                                        (random.random()-.5)*2,
                                        (random.random()-.5)*2,
                                        (random.random()-.5)*2]))
    

learningRate = 0.5
epochs = 500

TotalErrorArray = []

sameErrorCount = 0
previousError = 0

for epoch in range(epochs):
    TotalError = 0
    
    for row in range(750):
        
        #Forward propagation
        #Hidden neurons
        z1negIj = -1 * numpy.dot(Data[row,:], n1Weight)
        z1 = 1/(1 + numpy.exp(z1negIj))
        
        z2negIj = -1 * numpy.dot(Data[row,:], n2Weight)
        z2 = 1/(1 + numpy.exp(z2negIj))
        
        z3negIj = -1 * numpy.dot(Data[row,:], n3Weight)
        z3 = 1/(1 + numpy.exp(z3negIj))
        
        z4negIj = -1 * numpy.dot(Data[row,:], n4Weight)
        z4 = 1/(1 + numpy.exp(z4negIj))
        
        z5negIj = -1 * numpy.dot(Data[row,:], n5Weight)
        z5 = 1/(1 + numpy.exp(z5negIj))
        
        #Output neuron
        hiddenInput = [1, z1, z2, z3, z4, z5]       
        z6NegIj = -1 * numpy.dot(hiddenInput, n6Weight)
        z6 = 1/(1 + numpy.exp(z6NegIj))
        
        
        #Track Training Errors
        guess = round(z6)
        TotalError = TotalError + abs(classes[row] - guess)
        
        
        #Error propagation
        #Output Neuron Error
        err6 = z6 * (1 - z6) * (classes[row] - z6)
        
        #Hidden Neuron Errors
        err1 = z1 * (1 - z1) * (err6 * n6Weight[1])
        err2 = z2 * (1 - z2) * (err6 * n6Weight[2])
        err3 = z3 * (1 - z3) * (err6 * n6Weight[3])
        err4 = z4 * (1 - z4) * (err6 * n6Weight[4])
        err5 = z5 * (1 - z5) * (err6 * n6Weight[5])
        
        
        #Update weights
        #Output neuron weight
        deltaN6Weight = numpy.transpose(numpy.array([learningRate * err6 * 1,
                                                     learningRate * err6 * z1,
                                                     learningRate * err4 * z2,
                                                     learningRate * err4 * z3,
                                                     learningRate * err4 * z4,
                                                     learningRate * err4 * z5]))
        n6Weight = n6Weight + deltaN6Weight
        
        #Hidden neuron weights
        deltaN1Weight = []
        deltaN2Weight = []
        deltaN3Weight = []
        deltaN4Weight = []
        deltaN5Weight = []
        
        for column in range(numpy.size(Data, 1)):
            deltaN1Weight.append(learningRate * err1 * Data[row, column])
            deltaN2Weight.append(learningRate * err2 * Data[row, column])
            deltaN3Weight.append(learningRate * err3 * Data[row, column])
            deltaN4Weight.append(learningRate * err4 * Data[row, column])
            deltaN5Weight.append(learningRate * err5 * Data[row, column])
            
            
        n1Weight = n1Weight + numpy.transpose(numpy.array(deltaN1Weight))
        n2Weight = n2Weight + numpy.transpose(numpy.array(deltaN2Weight))
        n3Weight = n3Weight + numpy.transpose(numpy.array(deltaN3Weight))
        n4Weight = n4Weight + numpy.transpose(numpy.array(deltaN4Weight))
        n5Weight = n5Weight + numpy.transpose(numpy.array(deltaN5Weight))
        
    print(TotalError)
    TotalErrorArray.append(TotalError)
    
    #Training limiter code
    if abs(TotalError - previousError) == 0:
        sameErrorCount+=1
    else:
        sameErrorCount = 0
        
    if sameErrorCount >=10:
        break
    
    
    previousError = TotalError
      
matplotlib.pyplot.plot(TotalErrorArray)


#Classifying unused 250 data rows

TotalErrorClass0 = 0
TotalClass0 = 0
TotalErrorClass1 = 0
TotalClass1 = 0
for row in range(750, 1000):
    
    #Forward propagation
    #Hidden layer neurons
    z1negIj = -1 * numpy.dot(Data[row,:], n1Weight)
    z1 = 1/(1 + numpy.exp(z1negIj))
    
    z2negIj = -1 * numpy.dot(Data[row,:], n2Weight)
    z2 = 1/(1 + numpy.exp(z2negIj))
    
    z3negIj = -1 * numpy.dot(Data[row,:], n3Weight)
    z3 = 1/(1 + numpy.exp(z3negIj))
    
    z4negIj = -1 * numpy.dot(Data[row,:], n4Weight)
    z4 = 1/(1 + numpy.exp(z4negIj))
    
    z5negIj = -1 * numpy.dot(Data[row,:], n5Weight)
    z5 = 1/(1 + numpy.exp(z5negIj))
    
    #Output neuron
    hiddenInput = [1, z1, z2, z3, z4, z5]       
    z6NegIj = -1 * numpy.dot(hiddenInput, n6Weight)
    z6 = 1/(1 + numpy.exp(z6NegIj))
    
    
    #Track Clasifying Errors
    guess = round(z6)
    
    if(classes[row] == 0):
        TotalClass0 += 1
    else:
        TotalClass1 += 1
    
    if(numpy.sign(classes[row] - guess) < 0):
        TotalErrorClass0 += 1
    if(numpy.sign(classes[row] - guess) > 0):
        TotalErrorClass1 += 1
    
print("\nTotal Classifying Errors for class 0: " + str(TotalErrorClass0) + "/" + str(TotalClass0))
print("Total Classifying Errors for class 1: " + str(TotalErrorClass1) + "/" + str(TotalClass1))  
print("Total Classifying Errors: " + str(TotalErrorClass0 + TotalErrorClass1) + "/" + str(TotalClass0 + TotalClass1))  
        
