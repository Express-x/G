import numpy as np
from layers import FCLayer, ActivationLayer, InputLayer, HiddenLayer, OutputLayer, DenseLayer, ReceptronLayer, tanh, tanh_prime, sigmoid, sigmoid_prime, relu, relu_prime
from norms import LayerNorm, AttentionNorm, r1_loss, r2_loss, autograd_descent, greedy_gradient_descent, time_step_gradient_descent
from loss_functions import mse, mae, binary_cross_entropy, categorical_cross_entropy, hinge_loss, squared_hinge_loss, huber_loss, kl_divergence, cosine_similarity_loss


# Test cases for layers
X = np.array([[0.1, 0.2, 0.3]])

# Test FCLayer
fc_layer = FCLayer(3, 2)
output = fc_layer.forward_propagation(X)
print("FCLayer Output:", output)

# Test ActivationLayer (tanh)
activation_layer_tanh = ActivationLayer(tanh, tanh_prime)
output = activation_layer_tanh.forward_propagation(X)
print("ActivationLayer (tanh) Output:", output)

# Test InputLayer
input_layer = InputLayer(3)
output = input_layer.forward_propagation(X)
print("InputLayer Output:", output)

# Test HiddenLayer
hidden_layer = HiddenLayer(3, 4)
output = hidden_layer.forward_propagation(X)
print("HiddenLayer Output:", output)

# Test OutputLayer
output_layer = OutputLayer(4, 2)
output = output_layer.forward_propagation(hidden_layer.forward_propagation(X))
print("OutputLayer Output:", output)

# Test DenseLayer
dense_layer = DenseLayer(3, 5)
output = dense_layer.forward_propagation(X)
print("DenseLayer Output:", output)

# Test ReceptronLayer
receptron_layer = ReceptronLayer(3)
output = receptron_layer.forward_propagation(X)
print("ReceptronLayer Output:", output)

# Test cases for norms
X_norm = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])

# Test LayerNorm
layer_norm = LayerNorm(3)
output = layer_norm(X_norm)
print("LayerNorm Output:", output)

# Test AttentionNorm
attention_norm = AttentionNorm(3)
output = attention_norm(X_norm)
print("AttentionNorm Output:", output)

# Test r1_loss (requires gradients, using simple example)
real_pred = np.array([0.8, 0.9])
real_img = np.array([[0.1, 0.2], [0.3, 0.4]])
loss_r1 = r1_loss(real_pred, real_img)  # Fix: Remove extra argument
print("r1_loss Output:", loss_r1)

# Test r2_loss (requires gradients, using simple example)
fake_pred = np.array([0.2, 0.3])
fake_img = np.array([[0.5, 0.6], [0.7, 0.8]])
loss_r2 = r2_loss(fake_pred, fake_img)  # Fix: Remove extra argument
print("r2_loss Output:", loss_r2)

# Test cases for loss functions
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])

# Test MSE
loss_mse = mse(y_true, y_pred)
print("MSE Loss:", loss_mse)

# Test MAE
loss_mae = mae(y_true, y_pred)
print("MAE Loss:", loss_mae)

# Test Binary Cross-Entropy
loss_bce = binary_cross_entropy(y_true, y_pred)
print("Binary Cross-Entropy Loss:", loss_bce)

# Test Categorical Cross-Entropy (requires one-hot encoded y_true)
y_true_cce = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
y_pred_cce = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])
loss_cce = categorical_cross_entropy(y_true_cce, y_pred_cce)
print("Categorical Cross-Entropy Loss:", loss_cce)

# Test Hinge Loss
loss_hinge = hinge_loss(y_true, y_pred)
print("Hinge Loss:", loss_hinge)

# Test Squared Hinge Loss
loss_squared_hinge = squared_hinge_loss(y_true, y_pred)
print("Squared Hinge Loss:", loss_squared_hinge)

# Test Huber Loss
loss_huber = huber_loss(y_true, y_pred)
print("Huber Loss:", loss_huber)

# Test KL Divergence
y_true_kl = np.array([0.2, 0.3, 0.4, 0.1])
y_pred_kl = np.array([0.25, 0.25, 0.25, 0.25])
loss_kl = kl_divergence(y_true_kl, y_pred_kl)
print("KL Divergence Loss:", loss_kl)

# Test Cosine Similarity Loss
y_true_cosine = np.array([0.5, 0.5])
y_pred_cosine = np.array([0.8, 0.6])
loss_cosine = cosine_similarity_loss(y_true_cosine, y_pred_cosine)
print("Cosine Similarity Loss:", loss_cosine)


# Example usage of gradient descent functions (requires a model)
# Assuming you have a model defined with layers, forward, backward, and loss methods
# Example Model (replace with your actual model)
class ExampleModel:
    def __init__(self):
        self.layers = [
            FCLayer(3, 4),
            ActivationLayer(sigmoid, sigmoid_prime),
            OutputLayer(4, 1)
        ]

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def backward(self, loss):
        error = loss
        gradients = []
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate=0.1)  # Remove extra argument
            if isinstance(layer, (FCLayer, DenseLayer, OutputLayer)):
                gradients.append({"weights": layer.weights, "bias": layer.bias})
        return gradients[::-1]  # Reverse to match layer order

    def loss(self, y_pred, y_true):
        return mse(y_true, y_pred)


# Example data
X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
y_train = np.array([[0.4], [0.7], [1.0]])

# Create and train the model using different gradient descent methods
model = ExampleModel()

print("Training with Autograd Descent:")
autograd_descent(model, X_train, y_train, epochs=100)

print("\
Training with Greedy Gradient Descent:")
greedy_gradient_descent(model, X_train, y_train, epochs=100)

print("\
Training with Time Step Gradient Descent:")
time_step_gradient_descent(model, X_train, y_train, epochs=100, time_steps=2)
