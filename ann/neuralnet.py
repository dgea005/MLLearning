__author__ = 'Daniel'

# want to create using class
# goal is to have option for dropout, etc.

class ann_2:
    """
    An artificial neural network (2 layer) object
    """
    def __init__(self, features, hl1_size, hl2_size, classes):
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

    # initialise hyperparameters