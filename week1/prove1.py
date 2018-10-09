from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

from sklearn import datasets
        
class HardCodedClassifier:
    def fit(self, data_x, data_y):
        return HardCodedModel()


class HardCodedModel:
    def predict(self, test_data):

        # Create an array that will be returned
        lol = []

        # See how big the test_data size is
        for x in test_data:
            #Add a hardcoded 0 to it
            lol.append(0)

        # Return that array
        return lol

        
iris = datasets.load_iris()

# # Get the iris data from a CSV file
# iris_data = genfromtxt('iris.data', delimiter=",", dtype=None)

# # To test the data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7)

## An actual machine learning model
# classifier = GaussianNB()
# model = classifier.fit(x_train, y_train)
# targets_predicted = model.predict(x_test)

## My hardcoded model
classifier = HardCodedClassifier()
model = classifier.fit(x_train, y_train)
targets_predicted = model.predict(x_test)

# Calculate the percent right
total = 0
wrong = 0

# Iterates through both lists to see how many we were able to predict
for x,y in zip(targets_predicted, y_test):
    total += 1
    if x != y:
        wrong += 1

# This is your score out of the total
actual = total - wrong

# Let's get a percentage
percent = (actual / total) * 100

# Print this baby out
print("Your percent accuracy is {}".format(round(percent, 2)))