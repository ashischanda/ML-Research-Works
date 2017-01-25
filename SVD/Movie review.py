from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import *
from scipy.stats.stats import pearsonr
from numpy import linalg as la
import numpy as np

data = np.array([ 
           #Ben, Tom, John, Fred
            [5,5,0,5], # season 1
            [5,0,3,4], # season 2
            [3,4,0,3], # season 3
            [0,0,5,3], # season 4
            [5,4,4,5], # season 5
            [5,4,5,5]  # season 6
            ])
samples, features = shape(data)
def svd(data, S=2):
    
    U, s, V = linalg.svd(data) # calculate SVD
   
    #taking season data:  our samples
    newdata = U[:, :S]

    # this line is used to retrieve dataset
    #Sig = mat(eye(S) * s[:S])
    # ~ new = U[:,:2]*Sig*V[:2,:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for i in range( samples):
        #print (newdata[i, 0]," ", newdata[i, 1], " ", raw_data[i, -1] )
        #raw_data[i, -1]  means last column item of original data    
        ax.scatter(newdata[i, 0], newdata[i, 1])
        ax.annotate( str(i+1) , ( newdata[i, 0], newdata[i, 1])   )
    
    plt.xlabel('SVD1')
    plt.ylabel('SVD2')
    plt.show()
    # Result: season 5 and 6 are almost same
    
    
    #taking user data:  our features
    V = np.transpose(V)     # we need to apply the transpose example
    newdata = V[:, :S]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for i in range( features):
        #print (newdata[i, 0]," ", newdata[i, 1], " ", raw_data[i, -1] )
        #raw_data[i, -1]  means last column item of original data    
        ax.scatter(newdata[i, 0], newdata[i, 1] )
        ax.annotate( str(i+1) , ( newdata[i, 0], newdata[i, 1])   )
    
    plt.xlabel('SVD1')
    plt.ylabel('SVD2')
    plt.show()
    #user 1 and 4 (Ben, Fred) are almost same. They have same taste

svd(data, 2)
#reference: https://www.igvita.com/2007/01/15/svd-recommendation-system-in-ruby/
