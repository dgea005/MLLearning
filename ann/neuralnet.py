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
                 epochs=10, batch_size=100, rho=0.99, eta=1e-6,
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

    def j(self, data, label_matrix):
        """
        :params labels should have shape (xxx,), i.e., vector with no additional shape info
        :params data should have shape (features, length same as labels)
        """
        m = label_matrix.shape[0]
        f_pass = self.forward_propagation(data)
        # f_pass['a4'] will be the predictions for given weights
        cost = log_loss(label_matrix, f_pass['a4'].T)
        regularisation = np.sum(self.w1[:, 1:]**2) + \
                         np.sum(self.w2[:, 1:]**2) + \
                         np.sum(self.w3[:, 1:]**2)
        regularisation *= (self.reg_penalty / (2 * m))
        return cost + regularisation

    def gradients(self, data, label_matrix):
        # should really create m and label matrix else where
        # is it possible to have a function to initialise data and labels
        m = label_matrix.shape[0]
        # call to this function is where dropout would be called
        f_pass = self.forward_propagation(data)
        bias1 = np.ones(data.T.shape[1]).reshape(1, data.T.shape[1])
        data = np.concatenate((bias1, data.T), axis=0)
        d4 = f_pass['a4'].T - label_matrix
        d3 = np.dot(d4, self.w3) * self.f_prime(f_pass['a3']).T
        d3 = d3[:,1::]
        d2 = np.dot(d3, self.w2) * self.f_prime(f_pass['a2']).T
        d2 = d2[:,1::]
        D1 = (np.dot(data, d2).T) / m
        D2 = (np.dot(f_pass['a2'], d3).T) / m
        D3 = (np.dot(f_pass['a3'], d4).T) / m
        D1reg = np.zeros_like(D1)
        D1reg[:,1::] = (self.reg_penalty/m)* self.w1[:, 1::]
        D2reg = np.zeros_like(D2)
        D2reg[:,1::] = (self.reg_penalty/m) * self.w2[:, 1::]
        D3reg = np.zeros_like(D3)
        D3reg[:,1::] = (self.reg_penalty/m) * self.w3[:, 1::]
        D1 += D1reg
        D2 += D2reg
        D3 += D3reg
        # should return all three and then update params separately
        # will have to test this with gradient checking
        # so turn off regularization for now
        # should also change regularization to be L2
        return {'D1': D1, 'D2': D2, 'D3': D3}

    def train(self, data, labels):
        """ train records cost and epoch """
        m = len(labels)
        # initialise cost
        # need to create value matrix first
        # otherwise small batch sizes could not get correct sized matrix
        label_matrix = self.val_mat(labels)
        cost = self.j(data, label_matrix)
        # initialise accumulators
        D1_accum = 0
        D2_accum = 0
        D3_accum = 0
        w1_update_accum = 0
        w2_update_accum = 0
        w3_update_accum = 0
        iteration = 0
        for i in range(self.epochs):
            batch = np.random.choice(m, self.batch_size, replace=False).astype(int)
            # compute gradient
            gt = self.gradients(data[batch], label_matrix[batch])
            # this will return a dictionary of D1, D2, D3
            # i.e., gt['D1'], gt['D2'], gt['D3']
            # these will need to be applied to weight (w1, w2, w3) separately

            # accumulate gradient for each Delta
            # grad_accum = (rho * grad_accum) + ((1 - rho) * gt**2)

            D1_accum = (self.rho * D1_accum) + ((1 - self.rho) * gt['D1']**2)
            D2_accum = (self.rho * D2_accum) + ((1 - self.rho) * gt['D2']**2)
            D3_accum = (self.rho * D3_accum) + ((1 - self.rho) * gt['D3']**2)

            # compute update for each set of weights
            #  update = -( (np.sqrt(update_accum + self.eta) * gt) / (np.sqrt(grad_accum + self.eta)) )

            w1_update = -((np.sqrt(w1_update_accum + self.eta) * gt['D1']) / (np.sqrt(D1_accum + self.eta)))
            w2_update = -((np.sqrt(w2_update_accum + self.eta) * gt['D2']) / (np.sqrt(D2_accum + self.eta)))
            w3_update = -((np.sqrt(w3_update_accum + self.eta) * gt['D3']) / (np.sqrt(D3_accum + self.eta)))

            # accumulate update
            # update_accum = (rho * update_accum) + ((1 - rho)* update**2)

            w1_update_accum = (self.rho * w1_update_accum) + ((1 - self.rho)* w1_update**2)
            w2_update_accum = (self.rho * w2_update_accum) + ((1 - self.rho)* w2_update**2)
            w3_update_accum = (self.rho * w3_update_accum) + ((1 - self.rho)* w3_update**2)

            # apply update
            # nn_params += update
            self.w1 += w1_update
            self.w2 += w2_update
            self.w3 += w3_update

            # record cost + iterations
            cost_iterate = self.j(data, label_matrix)
            cost = np.append(cost, cost_iterate)
            iteration = np.append(iteration, i)
        return{'cost': cost, 'epoch': iteration}


