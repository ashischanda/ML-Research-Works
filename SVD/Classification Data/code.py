from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import *
from scipy.stats.stats import pearsonr
from numpy import linalg as la

# load data points
raw_data = loadtxt('iris_proc.data', delimiter=',')   
# if you have header, you can skip it: 
#loadtxt('iris_proc.data', delimiter=',', skiprows=1)

#database description: flower dataset has 5 columns
#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
#5. class:
#-- Iris Setosa
#-- Iris Versicolour
#-- Iris Virginica


samples, features = shape(raw_data)
data = mat(raw_data[:, :4])     # keeping data into a matrix format


def svd(data, S=2):
    
    U, s, V = linalg.svd(data) # calculate SVD
    # Umm, Smn, Vnn = Data mn  # dimension
    # But, Smn stores data in diagonal.
    # so, python returns the diagonal values in a single row
    
    print ( U.shape, s.shape, V.shape)
    #  (150,150) (4, )  (4, 4)
    
    # take out columns you don't need
    newdata = U[:, :S]

    # this line is used to retrieve dataset
    #Sig = mat(eye(S) * s[:S])
    # ~ new = U[:,:2]*Sig*V[:2,:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['blue', 'red', 'black']
    for i in range( samples):
        #print (newdata[i, 0]," ", newdata[i, 1], " ", raw_data[i, -1] )
        #raw_data[i, -1]  means last column item of original data    
        ax.scatter(newdata[i, 0], newdata[i, 1], color=colors[int(raw_data[i, -1])])
    
    plt.xlabel('SVD1')
    plt.ylabel('SVD2')
    plt.show()
    print (newdata.shape)  # (150, 2)

svd(data, 2)
