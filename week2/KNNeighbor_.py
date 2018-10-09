from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from operator import attrgetter
import numpy as np
import math

from sklearn import datasets

# To assist in combining the different datatypes for the predict function
class KnnObjectHolder:
    def __init__(self, data, target, distance):
        self.data = data
        self.target = target
        self.distance = distance        

class KNeighborsClassifier:

    def __init__(self, k):
        #How many neighbors are called upon
        self.k = k

    # To fit the data, all we are doing is saving it in an array
    # inside of the KNModel
    def fit(self, train, target):
        return KNModel(train, target, self.k)


class KNModel:
    def __init__(self, train, target, k):
        self.data = train
        self.target = target
        self.k = k

        
    # To predict all we do is calculate distances
    # This is NOT the most efficient way of doing things BUT it works.
    def predict(self, unclassified_data):

        # Different lists that are used to shift the data around
        distances = []
        distances_targets = []
        listKnnObjectHolder = []
        listPredictions = []

        # Let's see if we have a list of data
        
        # Loop through each element of data to be predicted
        for a in range(len(unclassified_data)):
            distances.clear()

            # Loop through each row of the data
            for n in range(len(self.data)):
                sum = 0

                # Loop through every element in the row n of self.data
                for p in range(len(self.data[n])):
                    sum += (unclassified_data[a][p] - self.data[n][p])**2
                
                distance = math.sqrt(sum)

                distances.append(distance)
                listKnnObjectHolder.append(KnnObjectHolder(self.data[n], self.target[n], distance))

                    #distances.append(distance)
            listKnnObjectHolder.sort(key=attrgetter('distance'))

            # To hold the distances of the closest k targets
            kClosestDistancesTargets = []

            for n in range(self.k):
                kClosestDistancesTargets.append(listKnnObjectHolder[n].target)

            listPredictions.append(max(set(kClosestDistancesTargets), key=kClosestDistancesTargets.count))

            # Reset the lists that will get used again
            listKnnObjectHolder.clear()
            kClosestDistancesTargets.clear()


        return listPredictions
        

iris = datasets.load_iris()

# # Get the iris data from a CSV file
# iris_data = genfromtxt('iris.data', delimiter=",", dtype=None)

# # To test the data/
x_train, y_test, x_target, y_target = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7)

## An actual machine learning model
# classifier = GaussianNB()
# model = classifier.fit(x_train, y_train)
# targets_predicted = model.predict(x_test)

# Preprocess the data to transform it to be more uniform
# This actually made my data decrease in accuracy
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_test = scaler.fit_transform(y_test)


## My hardcoded model
classifier = KNeighborsClassifier(6)
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
print("Your percent accuracy is {}%".format(round(percent, 2)))