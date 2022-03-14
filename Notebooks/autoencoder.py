import numpy as np
import random

#random.seed(123)
np.set_printoptions(precision=3)

# Simple sigmoid function that returns the sigmoid value for a given vector
def activation_function(x):
    return 1/(1 + np.exp(-x))


"""
A simple autoencoder to encode decimal numbers from 0-7 to a binary representation.
The architecture of the autoencoder is 8-3-8 (excl. the bias node of the hidden unit)
"""
class Autoencoder():
    def __init__(self):
        # Weight matrices
        self.theta1 = np.random.rand(3,8)
        self.theta2 = np.random.rand(8,4)

        # Activation
        self.a1 = np.zeros((8, 1))
        self.a2 = np.zeros((3, 1))
        self.a3 = np.zeros((8, 1))

        # Learning rate
        self.alpha = 0.01

    # Performs a forward pass through the network with the current weights
    def forward_propagation(self, x):
        self.a1 = x

        z2 = self.theta1.dot(self.a1)
        # Activations of hidden layer (encoded/binary representation of our number)
        self.a2 = np.vectorize(activation_function)(z2)
        # Add bias
        a2 = np.vstack((np.array(1), self.a2))

        #bias = np.ones(8)
        #theta2_with_bias = np.vstack((bias, self.theta2))
        z3 = self.theta2.dot(a2)
        # Output of the autoencoder = y
        self.a3 = np.vectorize(activation_function)(z3)
        return self.a3

    # Performs a forward pass through the network with the current weights
    def forward_prop(self, x, theta1, theta2):
        z2 = theta1.dot(x)
        # Activations of hidden layer (encoded/binary representation of our number)
        a2 = np.vectorize(activation_function)(z2)
        # Add bias
        a2 = np.vstack((np.array(1), a2))

        #bias = np.ones(8)
        #theta2_with_bias = np.vstack((bias, self.theta2))
        z3 = theta2.dot(a2)
        # Output of the autoencoder = y
        a3 = np.vectorize(activation_function)(z3)
        return a3

    def cost(self, h_theta, y):
        return np.sum(np.square(y- h_theta))/len(h_theta)

    def backpropagation(self, h_theta, y):
        delta_3 = h_theta - y
        # Ignore bias (In the formula there is theta^T)
        a2 = np.vstack((np.array(1), self.a2))
        delta_2 = (self.theta2.T.dot(delta_3)) * a2 * (1-a2)
        delta_2 = delta_2[1:,:]

        gradients_theta_1 = delta_2.dot(self.a1.T)
        gradients_theta_2 = delta_3.dot(a2.T)

        return gradients_theta_1, gradients_theta_2

    def gradient_checking(self, input, output, gradients_theta1, gradients_theta2):
        eps = 10E-5

        theta1_plus_eps = self.theta1 + eps
        h_plus_eps = self.forward_prop(input, theta1_plus_eps, self.theta2)
        theta1_minus_eps = self.theta1 - eps
        h_minus_eps = self.forward_prop(input, theta1_minus_eps, self.theta2)
        gradient_approx_theta1 = (self.cost(h_plus_eps, output) - self.cost(h_minus_eps, output)) / 2*eps
        #print(gradient_approx_theta1[i][j] - gradients_theta1[i][j])

        theta2_plus_eps = self.theta2 + eps
        h_plus_eps = self.forward_prop(input, self.theta1, theta2_plus_eps)
        theta2_minus_eps = self.theta2 - eps
        h_minus_eps = self.forward_prop(input, self.theta1, theta2_minus_eps)
        gradient_approx_theta2 = (self.cost(h_plus_eps, output) - self.cost(h_minus_eps, output)) / 2*eps

        return gradient_approx_theta1, gradient_approx_theta2

    def update_weights(self, gradients_theta_1, gradients_theta_2):
        self.theta1 = self.theta1 - self.alpha * gradients_theta_1
        self.theta2 = self.theta2 - self.alpha * gradients_theta_2

    def train(self, input, output):
        h_theta = self.forward_propagation(input)
        #print(self.cost(h_theta, input))
        gradients_theta_1, gradients_theta_2 = self.backpropagation(h_theta, output)
        #gradients_theta_1, gradients_theta_2 = self.gradient_checking(input, output, gradients_theta_1, gradients_theta_2)
        self.update_weights(gradients_theta_1, gradients_theta_2)

    # Input vector
inputs = [np.array([[0,0,0,0,0,0,0,1]]).T,
          np.array([[0,0,0,0,0,0,1,0]]).T,
          np.array([[0,0,0,0,0,1,0,0]]).T,
          np.array([[0,0,0,0,1,0,0,0]]).T,
          np.array([[0,0,0,1,0,0,0,0]]).T,
          np.array([[0,0,1,0,0,0,0,0]]).T,
          np.array([[0,1,0,0,0,0,0,0]]).T,
          np.array([[1,0,0,0,0,0,0,0]]).T]

a = Autoencoder()
for i in range(0,10000):
    input = inputs[random.randint(0,7)]
    a.train(input, input)

for input in inputs:
    forward = a.forward_propagation(input)
    print("Input", input.T, " Output:", forward.T)

print()
print("Theta1: \n", a.theta1)
print("Theta2: \n",a.theta2)
#y = np.array([0,0,0,0,0,0,0,1])
#h_theta = np.array([0.88771217, 0.91286169, 0.83737913, 0.82789999, 0.86225436, 0.83416626, 0.83667655, 0.84639284])
#cost = a.backpropagation(y, h_theta)

# Weight matrix L1
# Activation