__author__ = 'Daniel'

import numpy as np
import neuralnet as nn
import pylab as p

data = np.load("C:\\DataSets\\otto\dumps\\train_data")
labels = np.load("C:\\DataSets\\otto\\dumps\\train_labels")
# labels = labels.astype(int)

data = np.log10(data + 1)
# normalisation  (xi - mu) / sigma over all values
avg = np.mean(data)
sds = np.std(data)
# normalise training set
data -= avg
data /= sds

# testing neuralnet.py
network = nn.ann_2(features=93, hl1_size=400, hl2_size=400, classes=9)
network.set_hyperparams(epochs=300, batch_size=100)
tr = network.train(data, labels)
p.plot(tr["epoch"], tr["cost"])

print(network.accuracy(data, labels))
print(network.cross_entropy(data, labels))