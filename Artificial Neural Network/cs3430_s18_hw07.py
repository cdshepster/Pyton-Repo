#!/usr/bin/python

#####################################
# module: cs3430_s18_hw07.py
# COLE SHEPHERD
# A01895001
#####################################

import numpy as np
import pickle as cPickle

# sigmoid function
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def build_nn_wmats(mat_dims):
    matrix_list = []

    count = 0
    while count+1 < len(mat_dims):
        matrix_list.append(np.random.standard_normal(size=(mat_dims[count], mat_dims[count+1])))
        count += 1

    return matrix_list

def build_even_odd_nn():
    return build_nn_wmats((8, 4, 4, 2))

def build_231_nn():
    return build_nn_wmats((2, 3, 1))


def build_838_nn():
    return build_nn_wmats((8, 3, 8))


def create_nn_data():
    listOfBinaryString = []
    evenOddList = []

    for i in xrange(0, 129):
        num = format(i, '08b')
        numList = []
        for index in num:
            numList.append(int(index))
        listOfBinaryString.append(numList)

    for i in xrange(0, 129):
        if i % 2 == 0:
            evenOddList.append((1, 0))
        else:
            evenOddList.append((0, 1))

    return (np.array(listOfBinaryString, dtype=int), np.array(evenOddList, dtype=int))


def train_4_layer_nn(numIters, X, y, build):
    W1, W2, W3 = build()
    for j in range(numIters):
        a2 = sigmoid(np.dot(X, W1))
        a3 = sigmoid(np.dot(a2, W2))
        yHat = sigmoid(np.dot(a3, W3))

        yHat_error = y - yHat
        yHat_delta = yHat_error * sigmoid(yHat, deriv=True)

        a2_error = yHat_delta.dot(W3.T)
        a2_delta = a2_error * sigmoid(a3, deriv=True)
        
        a3_delta = a2_error * sigmoid(a2, deriv=True)

        W3 += a3.T.dot(yHat_delta)
        W1 += X.T.dot(a2_delta)
        W2 += X.T.dot(a3_delta)

    return W1, W2, W3


def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
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
    pass


def is_even_nn(n, wmats):
    if(wmats[3][n][0] == 1):
        return True
    else:
        return False


def eval_even_odd_nn(wmats):
    ## your code
    pass

# if __name__ == "__main__":
