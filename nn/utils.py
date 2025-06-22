import numpy as np

def compute_cost(Y, model):
    epsilon = 1e-8  # to prevent log(0)
    loss = Y * np.log(model + epsilon) + (1 - Y) * np.log(1 - model + epsilon)
    return -np.mean(loss)

def normalized(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)

    norm_X = (X-mu)/sigma
    return norm_X, mu, sigma