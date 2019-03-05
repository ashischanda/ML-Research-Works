# Matrix reduction using SVD
# We have a picture of Big matrix. 
# We want find a small matrix that will contain same picture quality
import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt

from scipy.misc import imread
im = imread("lenabw.jpg")       # Load an color image 
Pxx = im[:,:, 0]                # taking only one dimensional values

plt.imshow(Pxx)
#plt.plot(Pxx)                  # it will show the values in graph plotting
plt.show()                      # showing orginal image

print (Pxx.shape)  # original matrix shape: (554L, 507L)
U, s, Vh = SL.svd(Pxx, full_matrices=False)
assert np.allclose(Pxx, np.dot(U, np.dot( np.diag(s), Vh)  ) )
#Returns True if two arrays are element-wise equal within a tolerance

#old matrix size
print ("U ", U.shape )               #('U ', (554L, 507L))
print ("s ", s.shape )               #('s ', (507L,))
print ("Vh ", Vh.shape )             #('Vh ', (507L, 507L))


# print s and look, where the value change dramatically
# set new_dimension on the index
# print ( s[: new_dimension])

new_dimension = 30
s1 = [k for k in s]          # storing all s values for further checking
s[new_dimension:] = 0        # taking only first few elements and make all others zero 

new_a = np.dot(U, np.dot(np.diag(s), Vh)) # getting new image
plt.imshow(new_a)
plt.show()

# now reducing matrix size
U = U[:, :new_dimension]              # (554,           new_dimension)
s = s[:new_dimension]                 # (new_dimension, 1)
Vh = Vh [ :new_dimension, :]          # (new_dimension, 507)

new_a = np.dot(U, np.dot(np.diag(s), Vh))
print(new_a.shape )                   # (554, 507) #getting almost similar type matrix

plt.imshow(new_a)

'''
Look, at the beggining we had a matrix of (554L, 507L)
We have to remember (554 X 507) = 280,878 cells

Now, we have new U, s, Vh
We need to remember 554 * 30 = 16620
                    507 * 30 = 15210
                             = 30
          Total around       = 30,000 cells
          
          
But, if we need the old matrix, we can multiply new U, s, Vh and get new_a
------------------------
In this way, we can save space and keep same source of information
'''

