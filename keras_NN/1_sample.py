"""
Install: 
    1. theano
    2. conda install keras
    3. conda update dask
    
"""
# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/


from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]    # X is a numpy array. # X.shape (768, 8)
Y = dataset[:,8]      # Y is a numpy array. # Y.shape ( 768, 1)

# we are going to create 4 layer model. So, we are adding 3 layers with input layer
# (768 * 8), ( __ * 12), ( __ * 8), ( __ * 1)        # Final output: (768, 1)
# In matrix multiplication, column of A, row of B would be same. 
# So, we are only mentioning column number


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))    # Here, we don't need to mention row number
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#  we will use logarithmic loss, which for a binary classification problem is defined in Keras as “binary_crossentropy“
#  We will also use the efficient gradient descent algorithm “adam” 
#  because it is a classification problem, we will collect and report the classification accuracy as the metric.

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# we will run for a small number of iterations (150) and use a relatively small batch size of 10.
# We can also set the number of instances that are evaluated before a weight update in the network is performed, 
# called the batch size and set using the batch_size argument.


# evaluate the model
scores = model.evaluate(X, Y)       # You can use different matrix to evaluate
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print ("\nPrediction\n")

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
