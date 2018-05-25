#!/usr/bin/python

########################################################################
# module: ann.py
# author: vladimir kulyukin
# description: 3 simple ANNs
#
# Here is how you train and test an AND-Gate ANN:
#
# >>> AND_W1, AND_W2 = train_3_layer_nn(70000, X, y_and, build_2_3_1_nn)
# >>> AND_W1
# array([[ 5.08749222, -5.63290031,  3.12518918],
#       [-1.62952249,  1.93400938, -7.21289981]])
# >>> AND_W2
# array([[  5.88813025],
#       [ -7.85980301],
#       [-11.89541562]])
# >>> for x in X:
#	  print x, fit_3_layer_nn(x, AND_W1, AND_W2)
#	
# [0 0] [0]
# [1 0] [0]
# [0 1] [0]
# [1 1] [1]
#
# Here is how you train and test an OR-Gate ANN:
#
# >>> OR_W1, OR_W2 = train_3_layer_nn(70000, X, y_or, build_2_3_1_nn)
# >>> for x in X:
#	  print x, fit_3_layer_nn(x, OR_W1, OR_W2)
#	
# [0 0] [0]
# [1 0] [1]
# [0 1] [1]
# [1 1] [1]
#
# Here is how you train and test an ID ANN:
#
# >>> ID_W1, ID_W2 = train_3_layer_nn(70000, X1, y1, build_8_3_8_nn)
# >>> for x in X1:
#	  print x, fit_3_layer_nn(x, ID_W1, ID_W2)
#	
# [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]
# [0 1 0 0 0 0 0 0] [0 1 0 0 0 0 0 0]
# [0 0 1 0 0 0 0 0] [0 0 1 0 0 0 0 0]
# [0 0 0 1 0 0 0 0] [0 0 0 1 0 0 0 0]
# [0 0 0 0 1 0 0 0] [0 0 0 0 1 0 0 0]
# [0 0 0 0 0 1 0 0] [0 0 0 0 0 1 0 0]
# [0 0 0 0 0 0 1 0] [0 0 0 0 0 0 1 0]
# [0 0 0 0 0 0 0 1] [0 0 0 0 0 0 0 1]
#
########################################################################

import numpy as np

## ================ DATA ===================

X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

y_and = np.array([[0],
                  [0],
                  [0],
                  [1]])

y_or = np.array([[0],
                 [1],
                 [1],
                 [1]])

X1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])

y1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])

## ============== Designing and Training ==============

def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
        #return np.exp(-x)/((1 + np.exp(-x))**2)
    return 1 / (1 + np.exp(-x))

## Building 2 x 3 x 1 network
## W1 is the 2x3 matrix of synapse weights from the input layer to
## the hidden layer.
## W2 is the 3x1 matrix of synapse weights from the hidden layer
## to the output layer.
def build_2_3_1_nn():
    np.random.seed(1)
    W1 = np.random.randn(2, 3)
    W2 = np.random.randn(3, 1)
    return W1, W2

## Building 8 x 3 x 8 network
## W1 is the 8x3 matrix of synapse weights from the input layer to
## the hidden layer.
## W2 is the 3x8 matrix of synapse weights from the hidden layer
## to the output layer.
def build_8_3_8_nn():
    np.random.seed(1)
    W1 = np.random.randn(8, 3)
    W2 = np.random.randn(3, 8)
    return W1, W2

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    W1, W2 =  build()
    for j in range(numIters):
        Z2 = np.dot(X, W1)
        a2 = sigmoid(Z2)

        Z3 = np.dot(a2, W2)
        yHat = sigmoid(Z3)

        yHat_error = y - yHat
        yHat_delta = yHat_error * sigmoid(yHat, deriv=True)

        a2_error = yHat_delta.dot(W2.T)
        a2_delta = a2_error * sigmoid(a2, deriv=True)

        W2 += a2.T.dot(yHat_delta)
        W1 += X.T.dot(a2_delta)
    
    return W1, W2

## ============== Fitting ANN ==============

## x is the input vector to a 3-layer neural network
## whose synapse weights are defined by W1 and W2 matrices.
def fit_3_layer_nn(x, W1, W2, thresh=0.4, thresh_flag=True):
    a2 = sigmoid(np.dot(x, W1))
    yHat = sigmoid(np.dot(a2, W2))
    if thresh_flag == True:
        for y in np.nditer(yHat, op_flags=['readwrite']):
            if y > thresh:
                y[...] = 1
            else:
                y[...] = 0
        return yHat.astype(int)
    else:
        return yHat

