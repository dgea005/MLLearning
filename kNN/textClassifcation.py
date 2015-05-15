# Nearest Neighbor classifiers

import numpy as np
import os
import scipy.io
import random
from scipy.stats import mode
import numba
#import line_profiler
os.chdir("C:\\Users\\Daniel\\Dropbox\\Projects\\kNN")

import kNN as kn

# set seed
random.seed(1776)

# should move DataSet file to base at C
digData = scipy.io.loadmat("C:\\Users\\Daniel\\Documents\\DataSets\\text\\hwdigits.mat")

# test and train/val set splits (random)
# shuffle range(0, 5000) then index split as required

obs = np.arange(0, 5000)  # indices of all obs in order
np.random.shuffle(obs)    # shuffle all obs numbers


train_obs = obs[0:4200]
val_obs = obs[4200:4500]
test_obs = obs[4500::]

# take out of array
xDat = digData['X'] 
yDat = digData['y']

# create train set
xTrain = xDat[train_obs]
yTrain = yDat[train_obs]

# create val set
xVal = xDat[val_obs]
yVal = yDat[val_obs]

# create test set
xTest = xDat[test_obs]
yTest = yDat[test_obs]


# find hyperparameters that work best on validation set
val_accuracies = []
for k in [1,2,3,5,10,20,50,100]:
    pred = kn.kNN(xTrain,xVal,yTrain,yVal,k)
    acc = np.mean(pred == yVal.T)
    print("accuracy: %f" % (acc,))
    val_accuracies.append((k, acc))
val_accuracies
# tie between 2 and 3, will use 2 (simpler model is better)


# test with best working hyperparameters
# best is with k = 2
prediction = kn.kNN(xTrain, xTest, yTrain, yTest, k = 5)
np.mean(prediction == yTest.T)

# attempt at line profiling 
%load_ext line_profiler
%lprun -s -f kn.kNN -T lp_results.txt kn.kNN(xTrain, xTest, yTrain, yTest, k = 2)
%cat lp_results.txt

# spends 99.5% in distances
%timeit  kn.kNN(xTrain, xTest, yTrain, yTest, k = 2)

#In [6]: %timeit  kn.kNN(xTrain, xTest, yTrain, yTest, k = 2)
#1 loops, best of 3: 2.2 s per loop

# original time
#In [2]: %timeit  kn.kNN(xTrain, xTest, yTrain, yTest, k = 2)
#1 loops, best of 3: 20.4 s per loop

prediction = kn.kNNVec(xTrain, xTest, yTrain, yTest, 1)
np.mean(prediction == yTest.T)

%timeit  kn.kNNVec(xTrain, xTest, yTrain, yTest, k = 2)

#In [4]: %timeit  kn.kNNVec(xTrain, xTest, yTrain, yTest, k = 2)
#1 loops, best of 3: 8.02 s per loop



prediction = kn.kNNVecNumba(xTrain, xTest, yTrain, yTest, 1)
np.mean(prediction == yTest.T)


a = np.array([3, 4, 5, 2, 7, 1, 9, 6])
quicksort(a, start = 0, end = len(a)-1)
