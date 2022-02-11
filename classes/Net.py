import numpy as np
import math
from classes.Layer_Dense import Layer_Dense

class Net:
    def __init__(self, random_state):
        self.layers = []
        self.acc_history = None
        self.loss_history = None
        self.random_state = random_state
        
    def add_layer(self, n_inputs, n_outputs, activation):
        if activation == 'ReLU':
            activation = self.ReLU
        elif activation == 'sigmoid':
            activation = self.sigmoid
        elif activation == 'softmax':
            activation = self.softmax
        else:
            raise Exception(f"{activation} is not a valid activation function")
                            
        if len(self.layers) > 0:
            if n_inputs == self.layers[-1].n_outputs:
                layer = Layer_Dense(n_inputs, n_outputs, activation, self.random_state)
            else:
                raise Exception('Invalid layer dims')
        else:
            layer = Layer_Dense(n_inputs, n_outputs, activation)
            
        self.layers.append(layer)
                  
            
    def ReLU(self, Z, derive = False):
        if not derive:
            return np.maximum(Z, 0)
        else:
            return Z > 0
        
    def sigmoid(self, Z, derive = False):
        if not derive:
            return 1/(1 + np.exp(-Z))
        else:
            y = 1/(1 + np.exp(-Z))
            yp = y * (1.0 - y)
            return yp
    
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def one_hot(self, Y, n_classes):
        one_hot_Y = np.zeros((Y.size, n_classes))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def CCE(self, pred, label, m):
        loss = label * -np.log(pred)
        # Average over all samples in the batch
        loss = np.sum(np.sum(loss, axis = 1))/m
        return loss
    
    def update_params(self, layer, dW, db, learning_rate):
        layer.W -= learning_rate * dW
        layer.b -= learning_rate * db
        
    def forward(self, inputs):
        for i, layer in enumerate(self.layers):    
            layer.inputs = inputs
            Z = layer.W.dot(inputs) + layer.b
            A = layer.activation(Z)
            layer.Z = Z
            layer.A = A
            inputs = A
        return A
    
    def backward(self, outputs, Y, learning_rate, n_classes):
        label_one_hot = self.one_hot(Y, n_classes)  
        m = outputs.shape[1]
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                # dl/dz
                dZ = outputs - label_one_hot
                loss = self.CCE(outputs, label_one_hot, m)
            else:
                dZ = self.layers[i+1].W.T.dot(dZ) * layer.activation(layer.Z, derive = True)
            
            dW = 1/m * dZ.dot(layer.inputs.T)
            db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
            
            self.update_params(layer, dW, db, learning_rate)
        return loss
    
    def get_predictions(self, prediction):
        return np.argmax(prediction, 0)
    
    def score(self, X, Y):
        predictions = self.get_predictions(self.predict(X))
        return np.sum(predictions == Y) / Y.size
                
    def fit(self, X, Y, X_valid, y_valid, n_classes, epochs, batchsize = None,  learning_rate = 1):
        
        if not batchsize:
            batchsize = X.shape[1]
            
        n_iterations = abs(math.floor(-X.shape[1]/batchsize)) * epochs
        accuracy = np.zeros(n_iterations)    
        loss = np.zeros(n_iterations)
        
        iteration = 0
        for epoch in range(epochs):
            for start_idx in range(0, X.T.shape[0], batchsize):
                end_idx = min(start_idx + batchsize, X.T.shape[0])
                excerpt = slice(start_idx, end_idx)

                inputs = X.T[excerpt].T
                targets = Y.T[excerpt].T
            
                predictions = self.forward(inputs)
                
                l = self.backward(predictions, targets, learning_rate, n_classes)
                a = self.score(X_valid, y_valid)
                
                accuracy[iteration] = a
                loss[iteration] = l
                
                if iteration % 25 in [0,25]:
                    print(f'Epoch {epoch+1} Iteration {iteration}/{n_iterations}')
                    print(f'Accuracy: {round(a, 3)}')
                    print(f'Loss: {l}\n-')
                    
                iteration += 1
                
        self.acc_history = accuracy
        self.loss_history = loss
            
    def predict(self, X): 
        return self.forward(X)