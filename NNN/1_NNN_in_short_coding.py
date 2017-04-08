# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 01:29:55 2016

@author: Ashis Kumar Chanda
"""
# Code collected from a blog: A Neural Network in 13 lines of Python

## NNN structure: 3 input nodes, 1 output nodes. We have 4 training data
#input matrix: (4,3) => hidden layer (3,4) => hidden layer (4,1) => output (4,1)

import numpy as np
X = np.array([ [0,0,1],     # 4 by 3 input Matrix
              [0,1,1],
              [1,0,1],
              [1,1,1] ])
              
              
y = np.array([[0,1,1,0]]).T # 4 by 1 output Matrix

alpha,hidden_dim = (0.5,4)  # alpha = learning rate, hidden dimension state =4

synapse_0 = 2*np.random.random((3,hidden_dim)) - 1      #3 by 4 matrix of weight
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1      #4 by 1 matrix of weight

for j in range(6000):
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))          #sigmoid function
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
    
    # error finding
    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2)) # error * derivative
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
    
    #weight update    
    synapse_1 -= (alpha * layer_1.T.dot( layer_2_delta))
    synapse_0 -= (alpha * X.T.dot( layer_1_delta))

print (layer_2)

'''
output:
[[ 0.0231656 ]
 [ 0.97270372]
 [ 0.98623455]
 [ 0.02283812]]

#the output is same as our 'y'.
 '''
 
