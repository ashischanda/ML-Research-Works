import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# It has two parts. 1) data and 2)target
# diabetes.data, diabetes.target
# data is 442 by 10 values.   each row has 10 values
# target is 442 by 1. Only one single value in each row

selected_column = 3
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, selected_column] # here, we only consider one column
# now it has 442 by 1 values


#diabetes_X = diabetes.data[:, 0:4]     # but, you can set many columns


#So, we are looking one value, X and trying to guess another value, Y

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]  # avoiding last 20 rows
diabetes_X_test = diabetes_X[-20:]   # taking last 20 rows for test

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# **********************************************************************
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test) #diabetes_y_pred is set of values


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

print ( diabetes_y_test)
print ( diabetes_y_pred)
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.savefig("output_linear.png")

plt.show()
'''
output:
Coefficients: 
 [ 709.19471785]
Mean squared error: 4058.41
Variance score: 0.16
[ 233.   91.  111. ...,  132.  220.   57.]
[ 190.62399804  173.53233807  139.34901813 ...,  165.39345238  153.99901239    95.39903534]


'''
