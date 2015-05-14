__author__ = 'Daniel'

# want to create using class
# goal is to have option for dropout, etc.
import numpy as np
from sklearn.metrics import log_loss


class ann_2:
    """
    An artificial neural network (2 layer) object
    """
    def __init__(self, features, hl1_size, hl2_size, classes,
                 epochs=10, batch_size=100, rho=0.99, eta=0.1,
                 reg_penalty=1, eta_init = 0.1):
        """
        return a ann_2 object with these architectures
        :param features: number features input to network (input layer size)
        :param hl1_size: number of nodes in hidden layer 1
        :param hl2_size:  number of nodes in hidden layer 2
        :param classes: number of nodes in output layer
        :return:
        """
        self.features = features
        self.hl1_size = hl1_size
        self.hl2_size = hl2_size
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.rho = rho
        self.eta = eta
        self.reg_penalty = reg_penalty
        self.eta_init = eta_init
        # w1 = np.random.randn(self.hl1_size * (self.features + 1)) * np.sqrt(2.0/(self.hl1_size * (self.features + 1)))
        # w2 = np.random.randn(self.hl2_size * (self.hl1_size + 1)) * np.sqrt(2.0/(self.hl2_size * (self.hl1_size + 1)))
        # w3 = np.random.randn(self.classes * (self.hl2_size + 1)) * np.sqrt(2.0/(self.classes * (self.hl2_size + 1)))
        # do these need to have their matrix shapes
        self.w1 = np.random.normal(0, self.eta_init, (self.hl1_size, self.features + 1))
        self.w2 = np.random.normal(0, self.eta_init, (self.hl2_size, self.hl1_size + 1))
        self.w3 = np.random.normal(0, self.eta_init, (self.classes, self.hl2_size + 1))

    # initialise hyper parameters - method just to change the hyperparameters

    def set_hyperparams(self, epochs=100, batch_size=100, rho=0.99, eta=0.1, reg_penalty = 1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.rho = rho
        self.eta = eta
        self.reg_penalty = reg_penalty


    # is there some way to do this using numpy or numba/numexpr to make it faster
    @staticmethod
    def f(z):
        return z * (z > 0.)

    @staticmethod
    def f_prime(z):
        return z > 0.

    @staticmethod
    def val_mat(labels):
        m = len(labels)
        k = len(np.unique(labels))
        y_matrix = np.zeros((m, k))
        for i in range(m):
            y_matrix[i, labels[i]-1] = 1
        return y_matrix


    # won't include dropout at this stage
    def forward_propagation(self, data):
        # TODO make add_bias a function, pass in matrix to add bias to
        bias1 = np.ones(data.T.shape[1]).reshape(1, data.T.shape[1])
        data = np.concatenate((bias1, data.T), axis=0)
        z2 = np.dot(self.w1, data)
        # f will be our activation function - ReLU
        a2 = self.f(z2)
        # add the bias term
        bias2 = np.ones(a2.shape[1]).reshape(1, a2.shape[1])
        a2 = np.concatenate((bias2, a2), axis=0)
        z3 = np.dot(self.w2, a2)
        a3 = self.f(z3)
        bias3 = np.ones(a3.shape[1]).reshape(1, a3.shape[1])
        a3 = np.concatenate((bias3, a3), axis=0)
        z4 = np.dot(self.w3, a3)
        a4 = self.f(z4)
        return {'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3, 'z4': z4, 'a4': a4}


    # should have some function for checking the data and labels are in the correct dimension
    # this could be done at the train level

    def j(self, data, labels):
        """
        :params labels should have shape (xxx,), i.e., vector with no additional shape info
        :params data should have shape (features, length same as labels)
        """
        m = len(labels)
        f_pass = self.forward_propagation(data)
        # f_pass['a4'] will be the predictions for given weights
        label_matrix = self.val_mat(labels)
        # until this implementation is fixed ...
        # cost = ll.logloss(f_pass['a4'].T, label_matrix)
        cost = log_loss(label_matrix, f_pass['a4'].T)
        regularisation = np.sum(self.w1[:, 1:]**2) + \
                         np.sum(self.w2[:, 1:]**2) + \
                         np.sum(self.w3[:, 1:]**2)
        regularisation *= (self.reg_penalty / (2 * m))
        return cost + regularisation
