from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

from sklearn import datasets
        
class KNeighborsClassifier:

    def __init__(self, k):
        #How many neighbors are called upon
        self.k = k

    # To fit the data, all we are doing is saving it in an array
    # inside of the KNModel
    def fit(self, train, target):
        return KNModel(train, target)


class KNModel:
    def __init__(self, train, target):
        self.data = train
        self.target = target

        
    # To predict all we do is calculate distances
    def predict(self, unclassified_data):
        print(self.data)

        print(self.data[0])

        distances = []

        # Let's see if we have a list of data
        if len(unclassified_data > 1):
            # Loop through each element of data to be predicted
            for a in range(len(unclassified_data)):
                #print("LEEEETZ SEE: {}".format(a))
                #for b in unclassified_data[a]:
                #    print("Printing unclassified_data[{}]: {}".format(a, b)) 
                # Loop through all of the data and find the distance
                
                distances.clear()

                for n in range(len(self.data)):
                    print("Setting sum to 0")
                    sum = 0

                    print("row: {}".format(n))

                    print("unclassfied data: {}".format(unclassified_data[a]))
                    print(self.data[n])

                    for p in range(len(self.data[n])):
                        sum += (unclassified_data[a][p] - self.data[n][p])**2

                        #print("The sum is: {}".format(sum))
                        # p is going to loop through the individual numbers
                        # I need to do the distance between those and the predicted data
                    
                    #print("The sum outside of the loop is: {}".format(sum))
                    distance = math.sqrt(sum)

                    distances.append(distance)
                
                distances.sort()
                print("The size of the distances list is: {}".format(len(distances)))
                print("We found all the distances from {}".format(unclassified_data[a]))
                print("The list of distances are: {}".format(distances))
        else:
            print("You only have one data item, I need to implement this later")


iris = datasets.load_iris()

# # Get the iris data from a CSV file
# iris_data = genfromtxt('iris.data', delimiter=",", dtype=None)

# # To test the data/
x_train, y_test, x_target, y_target = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7)

## An actual machine learning model
# classifier = GaussianNB()
# model = classifier.fit(x_train, y_train)
# targets_predicted = model.predict(x_test)

## My hardcoded model
classifier = KNeighborsClassifier(3)
model = classifier.fit(x_train, x_target)
targets_predicted = model.predict(y_test)

# Calculate the percent right
total = 0
wrong = 0

# Iterates through both lists to see how many we were able to predict
for x,y in zip(targets_predicted, y_target):
    total += 1
    if x != y:
        wrong += 1

# This is your score out of the total
actual = total - wrong

# Let's get a percentage
percent = (actual / total) * 100

# Print this baby out
print("Your percent accuracy is {}".format(round(percent, 2)))