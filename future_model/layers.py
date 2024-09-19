import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size, weight_initializer='xavier', bias_initializer='zeros'):
        super().__init__()
        self.weights = self._initialize_weights(input_size, output_size, weight_initializer)
        self.bias = self._initialize_bias(output_size, bias_initializer)

    def _initialize_weights(self, input_size, output_size, initializer):
        if initializer == 'xavier':
            scale = np.sqrt(2 / (input_size + output_size))
            return np.random.normal(0, scale, (input_size, output_size))
        elif initializer == 'he':
            scale = np.sqrt(2 / input_size)
            return np.random.normal(0, scale, (input_size, output_size))
        else:
            return np.random.rand(input_size, output_size) - 0.5

    def _initialize_bias(self, output_size, initializer):
        if initializer == 'zeros':
            return np.zeros((1, output_size))
        else:
            return np.random.rand(1, output_size) - 0.5

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
        self.output = input_data
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error


class HiddenLayer(FCLayer):
    def __init__(self, input_size, output_size, weight_initializer='xavier', bias_initializer='zeros'):
        super().__init__(input_size, output_size, weight_initializer, bias_initializer)


class OutputLayer(FCLayer):
    def __init__(self, input_size, output_size, weight_initializer='xavier', bias_initializer='zeros'):
        super().__init__(input_size, output_size, weight_initializer, bias_initializer)


class DenseLayer(FCLayer):
    def __init__(self, input_size, output_size, weight_initializer='xavier', bias_initializer='zeros'):
        super().__init__(input_size, output_size, weight_initializer, bias_initializer)


class ReceptronLayer(FCLayer):
    def __init__(self, input_size, weight_initializer='xavier', bias_initializer='zeros'):
        super().__init__(input_size, 1, weight_initializer, bias_initializer)


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

class BatchNorm(Layer):
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))

        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))

    def forward_propagation(self, input_data, training=True):
        self.input = input_data

        if training:
            self.batch_mean = np.mean(input_data, axis=0, keepdims=True)
            self.batch_var = np.var(input_data, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

            self.x_hat = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        else:
            self.x_hat = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        self.output = self.gamma * self.x_hat + self.beta
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        batch_size = self.input.shape[0]

        dgamma = np.sum(output_error * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(output_error, axis=0, keepdims=True)

        dx_hat = output_error * self.gamma
        dvar = np.sum(dx_hat * (self.input - self.batch_mean) * (-0.5) * np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_hat * (-1) / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.input - self.batch_mean), axis=0, keepdims=True)

        dx = dx_hat / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.input - self.batch_mean) / batch_size + dmean / batch_size

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dx


class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, weight_initializer='xavier', bias_initializer='zeros'):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = self._initialize_weights(weight_initializer)
        self.bias = self._initialize_bias(bias_initializer)

    def _initialize_weights(self, initializer):
        if initializer == 'xavier':
            scale = np.sqrt(2 / (self.input_channels * self.kernel_size * self.kernel_size + self.output_channels))
            return np.random.normal(0, scale, (self.output_channels, self.input_channels, self.kernel_size, self.kernel_size))
        elif initializer == 'he':
            scale = np.sqrt(2 / (self.input_channels * self.kernel_size * self.kernel_size))
            return np.random.normal(0, scale, (self.output_channels, self.input_channels, self.kernel_size, self.kernel_size))
        else:
            return np.random.rand(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size) - 0.5

    def _initialize_bias(self, initializer):
        if initializer == 'zeros':
            return np.zeros((self.output_channels, 1))
        else:
            return np.random.rand(self.output_channels, 1) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        batch_size, _, input_height, input_width = input_data.shape

        output_height = int((input_height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        output_width = int((input_width + 2 * self.padding - self.kernel_size) / self.stride) + 1

        self.output = np.zeros((batch_size, self.output_channels, output_height, output_width))

        # Pad the input data
        input_padded = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        for i in range(output_height):
            for j in range(output_width):
                for k in range(self.output_channels):
                    receptive_field = input_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                    self.output[:, k, i, j] = np.sum(receptive_field * self.weights[k, :, :, :], axis=(1, 2, 3)) + self.bias[k]

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        batch_size, _, output_height, output_width = output_error.shape

        dweights = np.zeros_like(self.weights)
        dbias = np.zeros_like(self.bias)
        dinput_padded = np.zeros_like(np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant'))

        for i in range(output_height):
            for j in range(output_width):
                for k in range(self.output_channels):
                    receptive_field = self.input[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                    dweights[k, :, :, :] += np.sum(receptive_field * output_error[:, k, i, j][:, None, None, None], axis=0)
                    dbias[k] += np.sum(output_error[:, k, i, j], axis=0)
                    dinput_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += self.weights[k, :, :, :] * output_error[:, k, i, j][:, None, None, None]

        self.weights -= learning_rate * dweights
        self.bias -= learning_rate * dbias

        return dinput_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]


class MaxPool2D(Layer):
    def __init__(self, pool_size, stride=1):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data):
        self.input = input_data
        batch_size, input_channels, input_height, input_width = input_data.shape

        output_height = int((input_height - self.pool_size) / self.stride) + 1
        output_width = int((input_width - self.pool_size) / self.stride) + 1

        self.output = np.zeros((batch_size, input_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                receptive_field = input_data[:, :, i * self.stride:i * self.stride + self.pool_size, j * self.stride:j * self.stride + self.pool_size]
                self.output[:, :, i, j] = np.max(receptive_field, axis=(2, 3))

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        batch_size, input_channels, output_height, output_width = output_error.shape

        dinput = np.zeros_like(self.input)

        for i in range(output_height):
            for j in range(output_width):
                receptive_field = self.input[:, :, i * self.stride:i * self.stride + self.pool_size, j * self.stride:j * self.stride + self.pool_size]
                mask = (receptive_field == np.max(receptive_field, axis=(2, 3))[:, :, None, None])
                dinput[:, :, i * self.stride:i * self.stride + self.pool_size, j * self.stride:j * self.stride + self.pool_size] += mask * output_error[:, :, i, j][:, :, None, None]

        return dinput


class RNN(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward_propagation(self, input_data):
        """
        Forward propagation through time for a simple RNN.

        Args:
            input_data (numpy.ndarray): Input data of shape (sequence_length, input_size).

        Returns:
            numpy.ndarray: Output data of shape (sequence_length, output_size).
        """
        T, _ = input_data.shape
        h = np.zeros((T + 1, self.hidden_size, 1))  # Hidden state
        y = np.zeros((T, self.output_size, 1))  # Output

        for t in range(T):
            h[t] = np.tanh(np.dot(self.Wxh, input_data[t].reshape(-1, 1)) + np.dot(self.Whh, h[t - 1]) + self.bh)
            y[t] = np.dot(self.Why, h[t]) + self.by

        self.h = h
        self.output = y
        return y

    def backward_propagation(self, output_error, learning_rate):
        """
        Backpropagation through time for a simple RNN.

        Args:
            output_error (numpy.ndarray): Error gradients from the output layer, shape (sequence_length, output_size).
            learning_rate (float): Learning rate for parameter updates.

        Returns:
            numpy.ndarray: Gradients of the loss with respect to the input data, shape (sequence_length, input_size).
        """
        T, _ = output_error.shape
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(self.h[0])

        for t in reversed(range(T)):
            dy = output_error[t].reshape(-1, 1)
            dWhy += np.dot(dy, self.h[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dhraw = (1 - self.h[t] * self.h[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, self.input[t].reshape(1, -1))
            dWhh += np.dot(dhraw, self.h[t - 1].T)
            dh_next = np.dot(self.Whh.T, dhraw)

        # Update parameters with gradient descent
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        return  # Return None, as the input data gradients are not needed for RNN


class Optimizers:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def apply_gradients(self, model, gradients):
        raise NotImplementedError


class SGD(Optimizers):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def apply_gradients(self, model, gradients):
        for i in range(len(model.layers)):
            if isinstance(model.layers[i], (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN)):
                model.layers[i].weights -= self.lr * gradients[i]['weights']
                model.layers[i].bias -= self.lr * gradients[i]['bias']


class Momentum(Optimizers):
    def __init__(self, lr=0.01, beta1=0.9):
        super().__init__(lr, beta1)
        self.m = None

    def apply_gradients(self, model, gradients):
        if self.m is None:
            self.m = [{"weights": np.zeros_like(layer.weights), "bias": np.zeros_like(layer.bias)}
                      for layer in model.layers if isinstance(layer, (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN))]

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN)):
                self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * gradients[i]['weights']
                self.m[i]['bias'] = self.beta1 * self.m[i]['bias'] + (1 - self.beta1) * gradients[i]['bias']
                model.layers[i].weights -= self.lr * self.m[i]['weights']
                model.layers[i].bias -= self.lr * self.m[i]['bias']


class RMSprop(Optimizers):
    def __init__(self, lr=0.001, beta2=0.999, epsilon=1e-8):
        super().__init__(lr, beta2=beta2, epsilon=epsilon)
        self.v = None

    def apply_gradients(self, model, gradients):
        if self.v is None:
            self.v = [{"weights": np.zeros_like(layer.weights), "bias": np.zeros_like(layer.bias)}
                      for layer in model.layers if isinstance(layer, (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN))]

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN)):
                self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * np.square(gradients[i]['weights'])
                self.v[i]['bias'] = self.beta2 * self.v[i]['bias'] + (1 - self.beta2) * np.square(gradients[i]['bias'])
                model.layers[i].weights -= self.lr * gradients[i]['weights'] / (np.sqrt(self.v[i]['weights']) + self.epsilon)
                model.layers[i].bias -= self.lr * gradients[i]['bias'] / (np.sqrt(self.v[i]['bias']) + self.epsilon)


class Adam(Optimizers):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr, beta1, beta2, epsilon)
        self.m = None
        self.v = None
        self.t = 0

    def apply_gradients(self, model, gradients):
        self.t += 1
        if self.m is None:
            self.m = [{"weights": np.zeros_like(layer.weights), "bias": np.zeros_like(layer.bias)}
                      for layer in model.layers if isinstance(layer, (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN))]
        if self.v is None:
            self.v = [{"weights": np.zeros_like(layer.weights), "bias": np.zeros_like(layer.bias)}
                      for layer in model.layers if isinstance(layer, (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN))]

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], (FCLayer, DenseLayer, OutputLayer, Conv2D, RNN)):
                self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * gradients[i]['weights']
                self.m[i]['bias'] = self.beta1 * self.m[i]['bias'] + (1 - self.beta1) * gradients[i]['bias']
                self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * np.square(gradients[i]['weights'])
                self.v[i]['bias'] = self.beta2 * self.v[i]['bias'] + (1 - self.beta2) * np.square(gradients[i]['bias'])

                m_corrected_w = self.m[i]['weights'] / (1 - np.power(self.beta1, self.t))
                v_corrected_w = self.v[i]['weights'] / (1 - np.power(self.beta2, self.t))
                m_corrected_b = self.m[i]['bias'] / (1 - np.power(self.beta1, self.t))
                v_corrected_b = self.v[i]['bias'] / (1 - np.power(self.beta2, self.t))

                model.layers[i].weights -= self.lr * m_corrected_w / (np.sqrt(v_corrected_w) + self.epsilon)
                model.layers[i].bias -= self.lr * m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon)


# Example of how to use the RNN layer
rnn_layer = RNN(input_size=10, hidden_size=20, output_size=5)
input_sequence = np.random.randn(15, 10)  # Example input sequence
output_sequence = rnn_layer.forward_propagation(input_sequence)
print("Output Sequence Shape:", output_sequence.shape)  # Expected shape: (15, 5, 1)
