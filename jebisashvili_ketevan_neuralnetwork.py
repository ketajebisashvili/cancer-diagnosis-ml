import matplotlib.pyplot as plt
import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred - y_true)/y_true.size

class FCHiddenLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = np.tanh(self.output)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        derivative = 1 - np.power(self.output, 2)
        error = output_error * derivative
        input_error = np.dot(error, self.weights.T)

        # Reshape the arrays to align the dimensions
        input_T = self.input.reshape(-1, 1)
        error_reshape = error.reshape(1, -1)

        weights_gradient = np.dot(input_T, error_reshape)
        bias_gradient = np.sum(error, axis=0, keepdims=True)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_error

class Network:
    def __init__(self, input_size):
        self.layers = []
        self.loss = None
        self.loss_prime = None

        # add two fully connected hidden layers with 2 neurons each
        self.layers.append(FCHiddenLayer(input_size, 2))
        self.layers.append(FCHiddenLayer(2, 2))

        # add final fully connected layer with 1 neuron
        self.layers.append(FCHiddenLayer(2, 1))

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output[0])  # Append the first element of the output for binary classification
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
