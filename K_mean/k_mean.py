#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:50:54 2017

@author: ashis
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import math


def getData():
    #data_point =  np.array( [ [2,3],[4,2],[1,2],[1,3], [6,4], [5,4], [4,5] ])
    #data_point =  np.array( [ [2,3],[4,2],[1,2],[1,3], [6,4], [5,4], [4,5], [5,6], [8,5] , [3,4], [1,5], [3,3] ])
    data_point =  np.array( [ [3,9],[4,8],[4,7],[2,8],[2,9],[3,10],[3,9],[3,8] ,[2,3],[4,2],[1,2],[1,3], [6,4], [5,4], [4,5], [5,6], [8,5] , [3,4], [1,5], [3,3] ])
    
    return data_point
    
# take a list of pairs (x,y)
def plotData(data_point):
    plt.plot(data_point[:,0], data_point[:,1], "o")
    # set the axis of graph.  (x1,x2  , y1,y2)
    plt.axis(( 0,xRange, 0, yRange))     
       
    
def plotResult(train_data, y_label, title):
    colors = ['blue', 'red', 'black', 'green']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for i in range( len(train_data) ):
         ax.scatter(train_data[i,0], train_data[i,1], color=colors[y_label[i] ])
    plt.axis(( 0,xRange, 0, yRange))  
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
      
    plt.show()
    
 
def initializeCenterPoints(K):
    x = np.random.randint(0, xRange)
    y = np.random.randint(0, yRange)
    return [x,y]
    
def getDistance( dataPointX, dataPointY, centroidX,centroidY ):
    value = math.sqrt(math.pow((centroidY - dataPointY), 2) + math.pow((centroidX - dataPointX), 2))
    return value
    
def updateCentroidPoint(train_data, y_label):
    #taking average of all assigned points
    
    #temX = np.zeros(K)  # it is a numpy array of zeros
    temX = [0] * K       # list of zeros
    temY = [0] * K
    temCount = [0] * K
    
    # suming all X, Y values for each cluster
    for i in range(0, len(train_data)):
        for j in range (0, K):
            if y_label[i] == j:
                temX[j] += train_data[i][0]
                temY[j] += train_data[i][1]
                temCount[j] +=1
    
    temValue = []
    for j in range(0, K):
        x = (temX[j]+1) / (temCount[j]+1)   # Here, +1 is considered to avoid divide by zero
        y = (temY[j]+1) / (temCount[j]+1) 
        tem = [x,y]
        temValue.append(tem)
        
    return temValue
    
    
def kMean(K, train_data):
    select_points = []
    y_label = np.random.randint(0, K, len( train_data) )
    
    for i in range(0,K):
        select_points.append( initializeCenterPoints(K) )
    
        iterationNum = 10
    while iterationNum>0 :
        
        for i in range(0, len(train_data)):
            min_distance = BIG_NUMBER
            for j in range(0, K):
                #print (train_data[i])
                #print (select_points[j])
                find_distance = getDistance( train_data[i][0], train_data[i][1], select_points[j][0],select_points[j][1] )
                if find_distance<min_distance:
                    min_distance = find_distance
                    y_label[i] = j      # assigning cluster number
                    
        # updating cluster points            
        select_points = updateCentroidPoint(train_data, y_label)
        iterationNum= iterationNum-1
        plotResult(train_data, y_label, "Iteration "+str(iterationNum) )
       
    return y_label
  
    
    
if __name__ == '__main__':
    global xRange, yRange, BIG_NUMBER
    xRange = 10
    yRange = 12
    
    BIG_NUMBER = 9999999
    
    train_data = getData()
    plotData(train_data)
    K = 2
    y_label = kMean(K, train_data)
    
    plotResult(train_data, y_label, "final result")
    
    

# drawing a line. vertical line from p1 = (70,100) to p2= (70, 250)
# plt.plot([70, 70], [100, 250], 'k-', lw=2)   #line width = 2
