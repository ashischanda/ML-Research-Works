import pandas
from keras.models import Sequential
from keras.layers import Dense

# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]


#It is a simple model that has a single fully connected hidden layer with the same number 
#of neurons as input attributes (13).
#The network uses good practices such as the rectifier activation function for the hidden layer. 
#No activation function is used for the output layer because it is a regression problem and 
#we are interested in predicting numerical values directly without transform.


def baseline_model(train_x, train_y, test_x, test_y):

    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y , epochs=150, batch_size=10)
    predictions = model.predict( test_x )
    
    return predictions

tem = X[:5]

p= baseline_model(X, Y, tem, Y[:5])
print (p)
print (Y[:5])
'''
output
[[ 30.906147  ]
 [ 23.55308914]
 [ 30.46029282]
 [ 31.54349327]
 [ 27.22394562]]
[ 24.   21.6  34.7  33.4  36.2]
'''
