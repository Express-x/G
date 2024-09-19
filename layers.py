import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class InputLayer(Layer):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data  # Identity function for input layer
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error  # No change needed for input layer


class HiddenLayer(FCLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)


class OutputLayer(FCLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)


class DenseLayer(FCLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)


class ReceptronLayer(FCLayer):
    def __init__(self, input_size):
        super().__init__(input_size, 1)


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

