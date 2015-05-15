# Nearest Neighbor classifier
import numpy as np
import numba


@numba.njit
def distancesL2(testM, trainM, xTrain, xTest, vals, temp, temp2):
    for j in range(testM):
        for i in range(trainM):
            np.subtract(xTrain[i], xTest[j], temp)
            np.multiply(temp, temp, temp2)
            vals[j, i] = np.sqrt(np.sum(temp2))
    return vals

# this uses L1 distance
@numba.njit
def distancesL1(testM, trainM, xTrain, xTest, vals, temp, temp2):
    for j in range(testM):
        for i in range(trainM):
            np.subtract(xTrain[i], xTest[j], temp)
            np.absolute(temp, temp2)
            vals[j, i] = np.sum(temp2)
    return vals


def kNN(xTrain, xTest, yTrain, yTest, k = 1):
    '''
    xTrain/yTrain: has the shape (observations, features/variables)
    xTest/yTest: has the shape (observations, 1) - i.e., column vector
    k: is the number of lowest distances to choose prediction from
    returns prediction: highest vote of k lowest predictions 
    '''
    testM = len(yTest) # number of obs in test sample
    trainM = len(yTrain) # number of obs in train sample
    temp = np.zeros(len(xTrain[0]))
    temp2 = np.zeros(len(xTrain[0]))
    vals = np.zeros((testM, trainM), dtype = "float64")
    vals = distancesL2(testM, trainM, xTrain, xTest, vals, temp, temp2)
    # get indices of training set for smallest distances
    min_ind = np.zeros((testM, k), dtype = "int32")
    for i in range(0, testM):
        min_ind[i] = vals[i].argsort()[:k]
    # assign labels (from yTrain) using indices found above
    pred = np.zeros_like(min_ind)
    for i in range(0, testM):
        pred[i] = yTrain[min_ind[i]].T   # have to take tranpose since in column vec form
    # get most common classification from the k best distances
    # ( from stack exchange)
    axis = 1
    u, indices = np.unique(pred, return_inverse=True)
    predictions = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(pred.shape),
                                    None, np.max(indices) + 1), axis=axis)]
    return predictions









