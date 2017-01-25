import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt

from scipy.misc import imread
im = imread("lenabw.jpg") # Load an color image 
Pxx = im[:,:, 0]          # taking only one dimensional values

plt.imshow(Pxx)
#plt.plot(Pxx)  # it will show the values in graph plotting
plt.show()      # showing orginal image

U, s, Vh = SL.svd(Pxx, full_matrices=False)
assert np.allclose(Pxx, np.dot(U, np.dot( np.diag(s), Vh)  ) )
#Returns True if two arrays are element-wise equal within a tolerance

print ("s ", s.shape ) # 507 by 1
s1 = [k for k in s]    # storing all s values for further checking
s[8:] = 0  # taking only first few elements and make all others zero 

new_a = np.dot(U, np.dot(np.diag(s), Vh)) # getting new image
plt.imshow(new_a)
plt.show()

# now reducing matrix size
U = U[:, :8]            # (554, 8)
Vh = Vh [ :8, :]        # (8, 507)
s = s[:8]               # (8,1)
new_a = np.dot(U, np.dot(np.diag(s), Vh))
print(new_a.shape )  # (554, 507) #getting almost similar type matrix

plt.imshow(new_a)
