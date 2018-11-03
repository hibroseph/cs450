import numpy as np
import math

# Some constants that will be used but we can't have
# constants in python so I just put them as uppercase
sigmoid = np.vectorize(lambda x: 1/(1 + math.exp(-x)))


class NeuralNetwork:
    def __init__(self, input, target, layers):
        self.input = input
        self.target = target
        self.output = np.zeros(target.shape)

        # one array to hold all of the weights
        # weights[0] would return an array of all of the weights in the
        # first layer, weights[1] would return the weights in the second layer
        self.weights = []

        for x in range(len(layers) - 1):
            array = np.random.rand(layers[x], layers[x+1])
            self.weights.append(array)

        # Convert the normal array into a numpy array
        self.weights = np.array(self.weights)

        # print(self.weights)

    def feedforward(self, input):
        # print "self.weights[0]: \n", self.weights[0]

        # Do the input layer and the first weights
        self.layer = sigmoid(np.dot(input, self.weights[0]))
        # print "self.layer: \n", self.layer

        for x in range(1, len(self.weights)):
            self.layer = sigmoid(np.dot(self.layer, self.weights[x]))
            # print "self.layer ", x, " :\n", self.layer

        self.output = self.layer
        # print("output: ", self.output)

        # print("output[0][0]: ", self.output[0][0])
        # print("output[0][1]: ", self.output[0][1])

    def train(self, iterations):

        for x in range(iterations):
            self.feedforward(self.input)

            # simple cost function
            error = ((self.output - self.target)**2)

            # print "Error: \n", error, "\n"

            # implement back propagation and adjust weights

    def predict(self, x_test, y_test):

        # Feed the test data through the nn
        self.feedforward(x_test)
        
        # finding which rows are more
        answer = self.output[:,0] < self.output[:,1]
        # if true means the first column was bigger than the second column
        # if false means the opposite

        answerz = (answer == y_test)
        
        sumTrue = sum(answerz[:,0])

        accurarcy = sumTrue/float(x_test.shape[0]) * 100
        print "Accuracy: ", accurarcy, "%"

# x_train : data
# y_train : targets
# l : array depicting the amount of nodes in each layer including input
# eg. [3,2,3,2] means 3 nodes in input 2 nodes in layer 1,
# 3 nodes in layer 2 and 2 nodes in the output
# x_test : the data to be predicted
# Simple data format
x_train = np.array([[0, 1, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 0, 1],
                    [1, 0, 1, 1]])

# Simple target format
# This shows:
# [0, 1, 0, 1] : [0]
# [1, 1, 0, 1] : [1]
# [0, 1, 0, 1] : [0]
# [1, 0, 1, 1] : [1]
y_train = np.array([[0], [1], [0], [1]])

# x.shape[1] will return the amount of columns, ie the inputs
l = np.array([x_train.shape[1], 2, 3, 2])

# The test data
x_test = np.array([[0, 1, 0, 1],
                   [1, 1, 0, 1],
                   [1, 1, 1, 1],
                   [0, 0, 0, 1],
                   [1, 0, 0, 1]])

y_test = np.array([[1], [0], [1], [1], [1]])

# feed the data into the NeuralNetwork
nn = NeuralNetwork(x_train, y_train, l)

print "We are training"

nn.train(1)

print "We are predicting"
nn.predict(x_test, y_test)
