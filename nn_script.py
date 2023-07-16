
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### GETTING DATA
data = pd.read_csv("data_path")
data = np.array(data) # convert to array instead of lists to allow for management later
m, n = data.shape # m = number of rows, n = number of columns
np.random.shuffle(data) # shuffle before splitting into dev and training sets


### SPLITING DATA INTO TESTING AND TRAINING
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255 
# ^This step is to normalize image, i.e make sure pixel values are between 0 and 1
#this helps the model converge faster

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape


### INITIALIZE WEIGHTS AND BIASES 
def init_params():
    W1 = (np.random.rand(10, 784) - 0.5) * 0.01
    b1 = (np.random.rand(10, 1) - 0.5) * 0.01
    W2 = (np.random.rand(10, 10) - 0.5) * 0.01
    b2 = (np.random.rand(10, 1) - 0.5) * 0.01
    return W1, b1, W2, b2
# We subtract 0.5 to get a mix of positive and negative numbers
# Then * 0.01 because generally we don't want to initialise weights too large
# This helps ensure that the calculated gradients aren't too rediculously large, which can hurt convergence and stability

### DEFINING OUR KEY FUNCTIONS
def ReLU(Z):
    return np.maximum(Z, 0)
# Rectified Linear Unit is used 
# It introduces non linearity to the model (two linear componenst, but it is not continously linear, changes at 0)
# If we didn't use ReLU (or other activation functions like sigmoid), network layers would only be capable of perfrming linear transformations on the input data
# ReLU is generally preferred over sigmoid due to 1. simple function (no exponentials)
# and doesn't suffer from the gradient vanishin promelems - for large value, gradeint of sigmoid is near 0, potentially making learning very slow or stop
# however, similarly, in the potential cases where a neruon get negative inputs for all training instances, 
# we have a similar "dying ReLU" probel, where the neuron gets stuck and stops learning because the gradients are zero for all of the inputs (which are all negative)
# There are variations of ReLU which can solve this, and also some other hyperparameters and methods which can be tuned to help avoid this risk.

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
# Function is used to retrieve an output layer from a multi-class classification problem
# Output is in the form of probability-like values which help in making multi-class predictions
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
# Computes the output of the neural network given the inputs, weights and biases

def ReLU_deriv(Z):
    return Z > 0 # Neat way of returning derivative
# Returns boolean value of either False (0) or True (1),
# which are the derivates of the ReLU function!

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
# Used to convert the output labels into a matrix, typically full of 0s with 1 at the specific output index
# We add + 1 since python uses zero-based indexing
# 4 example: 0000100000
# og. label: 0123456789 

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
# Computes the gradient of the loss function wrt weights and biases
# This is used to update the parameteres (weights and biases) during gradient descent
# These are the correct derivatives, take them as is or there are many great articles explaining their derivations

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
# Here we update the parameters (weights and biases) based on the gradients computed in backprob and the learning rate (alpha)

def get_predictions(A2):
    return np.argmax(A2, 0)
# We use the output of the neural network (after softmax) to make the prediciton by selecting the class with the highest probability

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
# Compute proportion of the predictions that match with the true labels

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
# This puts together the key forward prob, back prob, and update params methods into a function we call gradient descent
# which performs iterative updates to the weights and biases of the network
# We also print the accuracy every 10 iterations to monitor performance and hopefully improvements


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
# Call the gradient descent method, inputing the training X and Y, learning rate, and iterations we want to compute
# We then also retrieve the returned optimized weightes W1 and W2 and biases b1 and b2, after the specified number of iterations.
# By retrieving these optimized paramters allows us to easily make predictions on new data (such as our testing data)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
# Here we make the prediction given the input X, using the trained weights and biases of the model
# We are only interest in retrieving A2, which represents the output probabilites of each class given the inputs
# With A2 we can run get_predictions which, as described above, makes its prediction by selecting the highest probability from A2
# The X input will represent our input data (X_train, see below)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
# Buidling a functions which allows us to test our predictions and visualise individuslly
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
# the 0, 1, 2, and 3 are the indexes from the training data we wish to see if our model correctly predicts


test_predictions = make_predictions(X_test, W1, b1, W2, b2)
# Running again, not on testing data (X_test)

get_accuracy(test_predictions, Y_test)
# ~ 80% - 90% accuracy on test data :)
