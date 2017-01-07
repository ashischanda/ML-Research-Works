# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:56:46 2016

@author: Ashis Kumar Chanda
"""

import numpy as np
import time
import random


Infinity = np.inf

def isMin(nearPoint, index, temp ):
   for i in range( len(nearPoint) ):
       if temp< nearPoint[i][1]:
           nearPoint[i][1] = temp
           nearPoint[i][0] = index
           break
   return nearPoint


def testknn(trainingX, trainingY, testingX, k):
   
  
   resultY = np.empty(  len( testingX), dtype=np.str ) # index, value 
   nearPoint = np.empty( [k, 2] ) # index, value
   
   for j in range(0, len( testingX) ):
       
       for i in range(k):
           nearPoint[i] = Infinity

       for i, singleData in enumerate( trainingX):
           temp = np.linalg.norm(singleData - testingX[j])
           nearPoint = isMin(nearPoint, i, temp)
        
       #index= nearPoint[0][0] 
       #resultY[j] = trainingY[index]
       
       # voting
       temCount = 0
       temValue = ""
       maxVote  = 0
       temIndex = 0
       winIndex = 0
       
       for p in range( len(nearPoint) ):
           temCount = 1 
           temIndex = nearPoint[p][0] 
           temValue = trainingY[temIndex]
           
           for q in range (p+1, len(nearPoint) ):
               if temValue == trainingY[ nearPoint[q][0] ]:
                   temCount = temCount + 1
                   
           if maxVote < temCount:
                maxVote = temCount
                winIndex = temIndex
                
        
       #print ("j value", j)
       #print ("winIndex value", winIndex)
       resultY[j] = trainingY[winIndex]
     
       
       if j>3:
           break
   print ("End")
   return resultY
   
def condensedata(trainingX, trainingY):
    randomNumber = int( random.random() *10 )
    
    subset = np.empty( [2000, 16] ) # index, value
    subsetLevel = np.empty( [2000, 1],  dtype=np.str  ) # index, value
    # take two random number to two different levels
    
    subset[0] = trainingX[randomNumber]
    subsetLevel[0] = trainingY[randomNumber]
    subset[1] = trainingX[randomNumber]
    subsetLevel[1] = trainingY[randomNumber]

    
    
    subsetElement =2
    minDistance = Infinity
    temLevel = 0
            
            
    for i, singleData in enumerate( trainingX):
        minDistance = Infinity
        for j in range (0, subsetElement):

             dist = np.linalg.norm(singleData - subset[j])
           
             if dist < minDistance:     # taking min value to compare for next decision
                 minDistance = dist
                 temLevel =j
                 
        if subsetLevel[temLevel][0] != trainingY[ i ]: # if not same, then add it subset
           subset[j] = trainingX[i]
           subsetLevel[j] = trainingY[ i ]
           #print (" i: ", i , " j ", j, " tem: ", temLevel, " ", subsetLevel[temLevel][0] ," ", trainingY[ i ] )
           subsetElement = subsetElement +1     # increasing subset element
        
        if subsetElement > 2000:        # Just taking 2000 data
            break;
    
    
    return subset, subsetLevel


def setValues():
    global k
    global nTrain
    global nTest, trainX, trainY, testX, testY
    
    inputRough = input('Enter value of k: ')
    k = int(inputRough)
    inputRough = input('Enter value of nTrain: ')
    nTrain= int(inputRough)
    inputRough = input('Enter value of nTest: ')
    nTest  = int(inputRough)
    print ("Your program will start now. Please, wait")
    
    dataset = np.loadtxt('letter-recognition.data', delimiter=',',  dtype='S')
    trainX= dataset[0:(nTrain+1)    ,1:].astype(np.float)
    testX = dataset[(nTrain+1): ((nTrain+1)+ nTest)     ,1:].astype(np.float)
    trainY = dataset[0: (nTrain+1)  ,0].astype(np.str)  
    testY =  dataset[(nTrain+1): ((nTrain+1)+ nTest)   ,0].astype(np.str)

setValues()


#nTrain = 15000       # = number of training examples
#nTest  = 5000             # = number of training examples
D   = 16    
#k = 3


timeStart = time.clock()
testY = testknn(trainX, trainY, testX, k)
#print (testY[0])

condensedIdx = condensedata(trainX, trainY)

print ("find values: \n", condensedIdx[0])
print ("find values: \n", condensedIdx[1])
print ( time.clock() - timeStart, "seconds\n" )
print ("\nProgram End\n")
