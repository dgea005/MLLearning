__author__ = 'Daniel'

import numpy as np
import neuralnet as nn

data = np.load("C:\\DataSets\\otto\dumps\\train_data")
labels = np.load("C:\\DataSets\\otto\\dumps\\train_labels")


# testing neuralnet.py
network = nn.ann_2(features=50, hl1_size=20, hl2_size=20, classes=5)
network.set_hyperparams(epochs=100)

