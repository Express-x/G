import numpy as np

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = np.ones(normalized_shape)
            self.bias = np.zeros(normalized_shape)

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        out = (x - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class AttentionNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim)

    def __call__(self, x):
        # x: [batch_size, seq_len, dim]
        mean = np.mean(x, axis=1, keepdims=True)  # [batch_size, 1, dim]
        var = np.var(x, axis=1, keepdims=True)  # [batch_size, 1, dim]
        out = (x - mean) / np.sqrt(var + self.eps)  # [batch_size, seq_len, dim]
        out = out * self.weight  # [batch_size, seq_len, dim]
        return out


def r1_loss(real_pred, real_img):
    grad_real = np.gradient(real_pred.sum(), real_img)
    grad_penalty = (np.linalg.norm(grad_real.reshape(grad_real.shape[0], -1), axis=1) ** 2).mean()
    return grad_penalty


def r2_loss(fake_pred, fake_img):
    grad_fake = np.gradient(fake_pred.sum(), fake_img)
    grad_penalty = (np.linalg.norm(grad_fake.reshape(grad_fake.shape[0], -1), axis=1) ** 2).mean()
    return grad_penalty


def autograd_descent(model, X, y, learning_rate=0.01, epochs=100):
    """Autograd (automatic differentiation) based Gradient Descent."""
    for epoch in range(epochs):
        # Forward propagation
        y_pred = model.forward(X)
        loss = model.loss(y_pred, y)

        # Backward propagation (using autograd for gradients)
        gradients = model.backward(loss)

        # Update weights and biases
        for i in range(len(model.layers)):
            if isinstance(model.layers[i], (FCLayer, DenseLayer, OutputLayer)):
                model.layers[i].weights -= learning_rate * gradients[i]["weights"]
                model.layers[i].bias -= learning_rate * gradients[i]["bias"]

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")


def greedy_gradient_descent(model, X, y, learning_rate=0.01, epochs=100):
    """Greedy Gradient Descent (updates weights after each data point)."""
    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward propagation
            y_pred = model.forward(X[i].reshape(1, -1))
            loss = model.loss(y_pred, y[i].reshape(1, -1))

            # Backward propagation
            gradients = model.backward(loss)

            # Update weights and biases
            for j in range(len(model.layers)):
                if isinstance(model.layers[j], (FCLayer, DenseLayer, OutputLayer)):
                    model.layers[j].weights -= learning_rate * gradients[j]["weights"]
                    model.layers[j].bias -= learning_rate * gradients[j]["bias"]

        if epoch % 10 == 0:
            y_pred = model.forward(X)
            loss = model.loss(y_pred, y)
            print(f"Epoch: {epoch}, Loss: {loss}")


def time_step_gradient_descent(model, X, y, learning_rate=0.01, epochs=100, time_steps=10):
    """Time Step Gradient Descent (updates weights every 'time_steps' data points)."""
    for epoch in range(epochs):
        for i in range(0, len(X), time_steps):
            # Forward propagation for a batch of time steps
            X_batch = X[i:i+time_steps]
            y_batch = y[i:i+time_steps]
            y_pred = model.forward(X_batch)
            loss = model.loss(y_pred, y_batch)

            # Backward propagation
            gradients = model.backward(loss)

            # Update weights and biases
            for j in range(len(model.layers)):
                if isinstance(model.layers[j], (FCLayer, DenseLayer, OutputLayer)):
                    model.layers[j].weights -= learning_rate * np.mean(gradients[j]["weights"], axis=0)
                    model.layers[j].bias -= learning_rate * np.mean(gradients[j]["bias"], axis=0)

        if epoch % 10 == 0:
            y_pred = model.forward(X)
            loss = model.loss(y_pred, y)
            print(f"Epoch: {epoch}, Loss: {loss}")
