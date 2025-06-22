import numpy as np
from .activations import ReLU, sigmoid
from .utils import compute_cost

def forward(X, W, B, layers):
    """
    Forward pass through the neural network.
    """
    l = len(layers)
    Z = [0.] * l
    A = [0.] * (l+1) #[X, A1, A2]
    A[0] = X
    for i in range(l):
        Z[i] = A[i] @ W[i] + B[i]
        if layers[i] == "ReLU":
            A[i+1] = ReLU(Z[i])
        elif layers[i] == "Sigmoid":
            A[i+1] = sigmoid(Z[i])
        elif layers[i] == "Linear":
            A[i+1] = Z[i]
    return A, Z

def backprop(layers, X, Y, A, B, W, Z):
    """"
    Backward pass through the neural network.
    To compute dJ_dW and dJ_dB which will be used to update the weights and biases."""
    m = X.shape[0]
    loss = compute_cost(Y, A[-1])
    dJ_dW = [0.]*len(layers)
    dJ_dB = [0.]*len(layers)
    dJ_dA = - (Y/A[-1] - (1-Y)/(1-A[-1]))
    for l in reversed(range(len(layers))):
        if layers[l] == "ReLU":
            dA_dZ = (Z[l] > 0).astype(float) 
        elif layers[l] == "Sigmoid":
            dA_dZ = sigmoid(Z[l]) * (1-sigmoid(Z[l]))
        elif layers[l] == "Linear":
            dA_dZ = np.ones_like(Z[l])
        dJ_dZ = dJ_dA * dA_dZ
        dJ_dC = dJ_dZ
        dJ_dB[l] = np.sum(dJ_dZ, axis=0, keepdims=True) / m
        dC_dA = W[l]; dC_dW = A[l]
        dJ_dA = dJ_dC @ W[l].T
        dJ_dW[l] = (dC_dW.T @ dJ_dC) / m 
    return dJ_dW, dJ_dB

def gradient_descent(X, Y, W_in, B_in, num_liters, alpha, layers):
    m, n = X.shape
    W = W_in #np.array() 1-D
    B = B_in #np.array() 1-D
    l = len(layers)
    A, Z = forward(X, W, B, layers)
    for i in range(num_liters):
        dJ_dW, dJ_dB = backprop(layers, X, Y, A, B, W, Z)
        for j in range(l):
            W[j] = W[j] - alpha * dJ_dW[j]
            B[j] = B[j] - alpha * dJ_dB[j]
        A, Z = forward(X, W, B, layers)
        J = compute_cost(Y, A[-1])
        if i % np.ceil(num_liters/10) == 0:
            print(f'At {i}, cost function(J) = {J}')
    return W, B