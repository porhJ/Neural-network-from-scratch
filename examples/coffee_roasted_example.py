"""
This is synthetic data for a coffee roasting process.
The dataset is from Andrew Ng's Machine Learning course on Coursera.
X is the input data, which includes roasting time and termperature.
Y is the output data, which includes the quality of the coffee; 0 = bad coffee, 1 = good coffee.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from nn.model import forward, gradient_descent
from nn.utils import normalized

def load_coffee_data():
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))

def prediction(y_hat):
    return (y_hat > 0.5).astype(int)

#Load the data
X, Y = load_coffee_data()

#Initialize the data
X_train = X[:160]; Y_train = Y[:160]
X_test = X[160:]; Y_test = Y[160:]
#Normalize the data
norm_X_train, mu, sigma = normalized(X_train)


#Planing the neural network
'''
input: X (200, 2)
Layer1 : 3 neurons, W_1 (2, 3) B_1 (3,) => output: A_1 (200, 3) ; activation = ReLU
Layer2 : 1 neurons, W_2 (3, 1) B_2 (1,) => output: A_2 (200, 1) ; activation = Sigmoid
'''

#Initialize the parameters
#As the plan, we will use 2 layers, so we will have 2 weight matrices and 2 bias vectors. 
W_1_in = np.random.randn(2, 3) * 0.01
B_1_in = np.zeros((3,))
W_2_in = np.random.randn(3, 1) * 0.01
B_2_in = np.zeros((1,))
W = [W_1_in, W_2_in]
B = [B_1_in, B_2_in]
num_liters = 10000
alpha = 1.0e-1

#initialize the layers
layers = ["ReLU", "Sigmoid"]

#Train the model
W_last, B_last = gradient_descent(norm_X_train, Y_train, W, B, num_liters, alpha, layers)

#Test the model
norm_X_test = (X_test - mu) / sigma
A, _ = forward(norm_X_test, W_last, B_last, layers) #A = [X, A_1,...., A_n], as n is the number of layers
#A[-1] is the output of the last layer, which is the prediction
Y_hat = A[-1]
#Compute the accuracy
accuracy = (prediction(Y_hat) == Y_test).mean()
accuracy*100 
print(f"Accuracy: {accuracy*100:.2f}%")