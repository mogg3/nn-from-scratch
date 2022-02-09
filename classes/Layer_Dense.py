import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_outputs, activation, random_state=0):
        np.random.seed(random_state)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = np.random.rand(n_outputs, n_inputs) - 0.5
        self.b = np.random.rand(n_outputs, 1) - 0.5

        self.activation = activation

        self.inputs = None
        self.Z = None
        self.A = None