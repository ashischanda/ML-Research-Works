# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:12:58 2016

@author: Ashis Kumar Chanda
"""
import numpy as np
import random
import time

def generateData(N):
    #taking two random points to draw line
    Xa =  random.uniform(-1, 1)
    Ya =  random.uniform(-1, 1)
    Xb =  random.uniform(-1, 1)
    Yb =  random.uniform(-1, 1)
    # line equation: (X - Xa) / (Xa-Xb) = (Y-Ya) / (Ya-Yb) 
    # take as: a X + b Y + c =0 format
    line =  np.array([Xb*Ya - Xa*Yb, Yb-Ya, Xa-Xb ])
    
    X= np.empty( [N, 2] , dtype= float)
    Y= np.empty( [N, 1], dtype= float)
    
    # evaluating output Yn
    for i in range (N):
        temX= random.uniform(-1, 1)
        temY= random.uniform(-1, 1)
        temPoint = np.array( [1, temX, temY])
        #chcking points with respect of line
        symbol = int ( np.sign( line.T.dot(temPoint) )  )
        
        Y[i]=  symbol
        X[i][0] = temX
        X[i][1] = temY
    
    return [X, Y]
    
#   *******************************************
def findError(w, X, Y):
    i= 0
    misPoints= []
    
    error_count =0
    N = len( Y)
    for x in range( len(X) ):
        temPoint = np.array( [1,  X[x][0], X[x][1] ]) # taking small letter x
        if int ( np.sign( w.T.dot( temPoint) )  ) != Y[i]:
            misPoints.append( (temPoint, Y[i]) )   #  append() takes exactly one argument. So, use ()            
            error_count =error_count+1      # counting 0/1 loss / error 
        
        i = i + 1
    
    
    if len(misPoints) ==0 :
        misX = np.zeros(3)
        misY = np.zeros(1)
    else:
        misX, misY = misPoints[ random.randrange(0,len(misPoints) ) ] # selecting one wrong point randomly 
    
    error = error_count/ float(N)           #taking average
    return error , misX, misY
    
#   *******************************************  
def pla(X, Y, w):
  
    flag = 1
    iteration = 1
    while (flag) : 
        errorValue, misX, misY = findError(w, X, Y)
        # misX, misY = taking a point randomly to update weight value
        
        if( errorValue == 0 ):  #checking error to stop the process
            break
        
        iteration = iteration +1    
      
        w = w+ ( misY * misX)	# updating weight values
    
    return w, iteration
        
    
#   *******************************************   
def pseudoinverse(X, Y):
    p = np.linalg.pinv( X)
    w =np.dot( p, Y )
    
    return w


if __name__ == '__main__':
    N = 10          # Set N value

    timeStart = time.clock()   
    # Algorithm 1:
    iters =0
    for i in range(100):
        a , b = generateData(N)
        w = np.zeros(3)     # w0 is initially zero.
        w, count = pla( a, b, w)
        iters = iters+ count
          
    print ("iteration : ", iters/100)     # taking average iteration
    print ("total time (with weight vector zero)\n")
    print ( ( time.clock() - timeStart) / 100 , "seconds\n" ) # taking average time
    
    timeStart = time.clock()
    
    # Algorithm 2:
    iters =0
    for i in range(100):
          a , b = generateData(N)
          w2 = pseudoinverse(a, b)
          w = np.zeros(3) 
          w[1]= w2[0]
          w[2]= w2[1]
          w, count = pla( a, b, w)
          iters = iters + count
        
    print ("iteration : ", iters/100)     # taking average iteration
    print ("total time (with weight)\n") 
    print ( ( time.clock() - timeStart) / 100 , "seconds\n" )
