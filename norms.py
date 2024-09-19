import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CUDA/GPU is available. Using CuPy for computations.")
except ImportError:
    GPU_AVAILABLE = False
    print("CUDA/GPU is not available. Using NumPy for computations.")

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
        if GPU_AVAILABLE:
            x = cp.asarray(x)  # Move data to GPU
            mean = cp.mean(x, axis=-1, keepdims=True)
            var = cp.var(x, axis=-1, keepdims=True)
            out = (x - mean) / cp.sqrt(var + self.eps)
            if self.elementwise_affine:
                out = out * self.weight + self.bias
            return cp.asnumpy(out)  # Move result back to CPU
        else:
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
        if GPU_AVAILABLE:
            x = cp.asarray(x)
            mean = cp.mean(x, axis=1, keepdims=True)
            var = cp.var(x, axis=1, keepdims=True)
            out = (x - mean) / cp.sqrt(var + self.eps)
            out = out * self.weight
            return cp.asnumpy(out)
        else:
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

from layers import BatchNorm

# Attention Mechanisms

class SelfAttention(Layer):
    def __init__(self, input_dim, dk, dv):
        super().__init__()
        self.dk = dk
        self.dv = dv

        self.Wq = self._init_weights((input_dim, dk))
        self.Wk = self._init_weights((input_dim, dk))
        self.Wv = self._init_weights((input_dim, dv))

    def _init_weights(self, shape):
        return np.random.randn(*shape) * 0.01

    def __call__(self, queries, keys, values):
        if GPU_AVAILABLE:
            queries, keys, values = cp.asarray(queries), cp.asarray(keys), cp.asarray(values)
            Q = cp.dot(queries, self.Wq)
            K = cp.dot(keys, self.Wk)
            V = cp.dot(values, self.Wv)

            scaled_attention = cp.matmul(Q, K.transpose(0, 2, 1)) / cp.sqrt(self.dk)
            attention_weights = cp.exp(scaled_attention) / cp.sum(cp.exp(scaled_attention), axis=-1, keepdims=True)
            output = cp.matmul(attention_weights, V)
            return cp.asnumpy(output)
        else:
            Q = np.dot(queries, self.Wq)
            K = np.dot(keys, self.Wk)
            V = np.dot(values, self.Wv)

            scaled_attention = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.dk)
            attention_weights = np.exp(scaled_attention) / np.sum(np.exp(scaled_attention), axis=-1, keepdims=True)
            output = np.matmul(attention_weights, V)
            return output


class MultiHeadAttention(Layer):
    def __init__(self, input_dim, dk, dv, num_heads):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads
        self.heads = [SelfAttention(input_dim, dk, dv) for _ in range(num_heads)]
        self.Wo = self._init_weights((num_heads * dv, input_dim))

    def _init_weights(self, shape):
        return np.random.randn(*shape) * 0.01

    def __call__(self, queries, keys, values):
        head_outputs = [head(queries, keys, values) for head in self.heads]
        concatenated_output = np.concatenate(head_outputs, axis=-1)
        if GPU_AVAILABLE:
            concatenated_output = cp.asarray(concatenated_output)
            output = cp.dot(concatenated_output, self.Wo)
            return cp.asnumpy(output)
        else:
            output = np.dot(concatenated_output, self.Wo)
            return output


class LinearAttention(Layer):
    def __init__(self, input_dim, dk):
        super().__init__()
        self.dk = dk
        self.Wq = self._init_weights((input_dim, dk))
        self.Wk = self._init_weights((input_dim, dk))
        self.Wv = self._init_weights((input_dim, dk))

    def _init_weights(self, shape):
        return np.random.randn(*shape) * 0.01

    def __call__(self, queries, keys, values):
        if GPU_AVAILABLE:
            queries, keys, values = cp.asarray(queries), cp.asarray(keys), cp.asarray(values)
            Q = cp.dot(queries, self.Wq)
            K = cp.dot(keys, self.Wk)
            V = cp.dot(values, self.Wv)

            scaled_attention = cp.matmul(Q, K.transpose(0, 2, 1)) / cp.sqrt(self.dk)
            attention_weights = cp.exp(scaled_attention - cp.max(scaled_attention, axis=-1, keepdims=True))
            attention_weights /= cp.sum(attention_weights, axis=-1, keepdims=True)
            output = cp.matmul(attention_weights, V)
            return cp.asnumpy(output)
        else:
            Q = np.dot(queries, self.Wq)
            K = np.dot(keys, self.Wk)
            V = np.dot(values, self.Wv)

            scaled_attention = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.dk)
            attention_weights = np.exp(scaled_attention - np.max(scaled_attention, axis=-1, keepdims=True))
            attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
            output = np.matmul(attention_weights, V)
            return output


class HierarchicalAttention(Layer):
    def __init__(self, input_dim, dk, dv, num_heads, hierarchy_levels):
        super().__init__()
        self.hierarchy_levels = hierarchy_levels
        self.multi_head_attention_layers = [
            MultiHeadAttention(input_dim, dk, dv, num_heads) for _ in range(hierarchy_levels)
        ]

    def __call__(self, queries, keys, values):
        for i in range(self.hierarchy_levels):
            if GPU_AVAILABLE:
                queries, keys, values = cp.asarray(queries), cp.asarray(keys), cp.asarray(values)
                attention_output = self.multi_head_attention_layers[i](queries, keys, values)
                queries = keys = values = attention_output  # Update queries, keys, values for the next level
                queries, keys, values = cp.asnumpy(queries), cp.asnumpy(keys), cp.asnumpy(values)
            else:
                attention_output = self.multi_head_attention_layers[i](queries, keys, values)
                queries = keys = values = attention_output  # Update queries, keys, values for the next level
        return attention_output
