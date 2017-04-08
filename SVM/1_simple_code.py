#  Support Vector Machine (SVM)     # It is a supervised process
#  Taking 6 points as input and try to find a line with Max margin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm  # sklearn is a ML package

X = np.array([[1,2],        # 6 points as input
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]    # Y labels of our input data

model = svm.SVC(kernel='linear', C = 1.0)   # calling linear kernel 
model.fit(X,y)    # Training our model

print("Predicting one point: class = ", model.predict([0.58,0.76]))
print("Predicting one point: class = ", model.predict([10.58,10.76]))

# visualize your data:
w = model.coef_[0]        #TAKING CO EFFICIENTS OF LINE
print(w)

a = -w[0] / w[1]    # SLOPE

xx = np.linspace(0,12)
yy = a * xx - model.intercept_[0] / w[1]
# AX + By + C = 0
# Y = (AX - C) / B 

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()
