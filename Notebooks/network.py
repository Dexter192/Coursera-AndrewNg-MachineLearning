import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],3) 
        self.weights2   = np.random.rand(4,8)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.biaslayer1 = np.c_[self.layer1, np.ones([8])] 
        self.output = sigmoid(np.dot(self.biaslayer1 , self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.biaslayer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1[:,:-1]
        self.weights2 += np.vstack([d_weights2, np.zeros(8)])

def equal_matrix(A,B):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if (A[i][j] != B[i][j]):
                return False
    return True

if __name__ == "__main__":
    X = np.identity(8)
    total_itr = 0
    min_itr = 100000000000000
    max_itr = 0
    num = 50
    for x in range(0, 1):
        nn = NeuralNetwork(X,X)
        iterations = 0
        while not equal_matrix(np.rint(nn.output), X):
            iterations += 1
            nn.feedforward()
            nn.backprop()
        total_itr += iterations
        min_itr = min(min_itr, iterations)
        max_itr = max(max_itr, iterations)
    total_itr = total_itr/num
    print("Converge after {} iterations, min {}, max {}".format(total_itr, min_itr, max_itr))    
    print(np.rint(nn.weights1))
    print(np.rint(nn.weights2))
    