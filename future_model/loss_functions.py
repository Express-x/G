import numpy as np

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# Mean Absolute Error (MAE)
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Binary Cross-Entropy (BCE)
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross-Entropy (CCE)
def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Hinge Loss (for SVM)
def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Squared Hinge Loss (for SVM)
def squared_hinge_loss(y_true, y_pred):
    return np.mean(np.power(np.maximum(0, 1 - y_true * y_pred), 2))

# Huber Loss (robust to outliers)
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    return np.mean(np.where(abs_error <= delta, 0.5 * np.power(error, 2), delta * (abs_error - 0.5 * delta)))

# Kullback-Leibler Divergence (KL Divergence)
def kl_divergence(y_true, y_pred):
    return np.sum(y_true * np.log(y_true / y_pred))

# Cosine Similarity Loss
def cosine_similarity_loss(y_true, y_pred):
    return 1 - np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
