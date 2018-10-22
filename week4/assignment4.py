from math import log
import pandas as pd
import statistics
import numpy as np

from sklearn.datasets import load_iris

# math.log(num,base)

class DTClassifer:
    def __init__(self):
        print("Created a Classifer")
    
    ###
    # This function takes in a whole dataset of training data (no targets)
    # and specifically finds the mean of each column and if the data at 
    # a specific point is less than the mean, it replaces it with a 0
    # otherwise a 1. Written by Joe Ridgley
    ###
    def binData(self, data_set):

        for x in data_set.columns:
            
            # Calculate mean of the column you are going down
            meanOfColumn = statistics.mean(data_set[:][x])
            
            for y in range(len(data_set)):
                # print(data_set[x][y])
                if data_set[x][y] < meanOfColumn:
                    data_set[x][y] = 0
                else:
                    data_set[x][y] = 1

        return data_set

    ###
    # This function takes in a column of the training data
    # and the training target and calculates the entropy of
    # the training data with respect to the training target.
    # Written by Jason Meziere
    ###
    def entropy(self,train_data,train_target):

        class_count = np.zeros((len(np.unique(train_target)) + 1,len(np.unique(train_data)) + 1))

        for i in range(len(train_data)):

            class_count[int(train_target[i]),int(train_data[i])] += 1

            class_count[int(train_target[i]),-1] += 1

            class_count[-1,int(train_data[i])] += 1

        # print(class_count)

        # print(len(class_count))

        # print(len(class_count[0]))

        entropy = 0

        for i in range(len(class_count)-1):

            for j in range(len(class_count[0])-1):

                # print(class_count[i,j])

                if class_count[i,j] != 0:

                    entropy += class_count[i,j] / class_count[-1,j]*(class_count[i,j]/class_count[-1,j]*log(class_count[i,j]/class_count[i,-1],2))

        return entropy


    # Fits the data to the model
    def fit(self, train, target):
        return Tree(train, target)

    ###
    # Finds the majority to return
    ###
    def findMajority(train):
        freq = {}

        # loop through all the training data to see what the most frequent is
        for train_index in train:
            if (freq.has_key(train_index)):
                freq[train_index] += 1
            else:
                freq[train_index] = 1

        max = 0
        major = ""

        for key in freq.keys():
            if freq[key]>max:
                max = freq[key]
                major = key
        
        return major

    ###
    # This function is responsble 
    ###
    def makeTree(train, target):
       train = train[:]
       target = target[:]

        # Find the majority to know the color
       majority = findMajority(train)


       tree = {majority:{}}

        ## This is has far as I got
       tree[majority][values]

       makeTree(train, target)

# Holds the root node to our tree
class Tree:
    def __init__(self, train, target):
        self.root = Node()

class Node(object):
    def __init__(self):
        print("Creating a node!")
        # A dictionary to hold other nodes when they get connected
        self.children = {}    
        # To tell if it is a leaf node or not, will be set automatically
        self.leafNode = False

    def __eq__(self, other):
        return 


## To Test my entropy
# The 1st digit represents number of attributes
# 2nd digit represents class 
creditScore = [[4, 3], [4, 2], [4,1]]
income = [[6, 5], [6, 1]]
collateral = [[6, 4], [6, 2]]
half = [[4,4]]

# calculate_entropy(half)

iris_data = pd.read_csv('iris_test.csv', header=None)

# Select all rows from columns 0 to 3
train_data = iris_data.loc[: , 0:3]
# Select all rows from column 4
target_data = iris_data.loc[: , 4]

train_data_colum = iris_data.loc[: , 0:0]

# calculate_new_entropy(train_data_colum, target_data)

classifer = DTClassifer()

binnedData = classifer.binData(train_data)

numpyDf = binnedData.values

numpyDfCol = numpyDf[: , 0]

classifer.entropy(numpyDfCol, target_data)
# print(numpyDfCol)
# print(binnedData)

# print(train_data)
# print(target_data)

# print(target)
# x = Node()
# y = Node()

# x.children["H"] = Node()