import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

#######################################################################
# This function calculates and displays the percentage numbers when 
# passed the predicted list and the key list.
#######################################################################
def calculate_display_percentage(predicted, key):
    # Calculate the percent right
        total = 0
        wrong = 0

        # Iterates through both lists to see how many we were able to predict
        for x,y in zip(predicted, key):
            total += 1
            if x != y:
                wrong += 1

        # This is your score out of the total
        actual = total - wrong

        # Let's get a percentage
        percent = (float(actual) / float(total)) * 100

        # Print this baby out
        print("Your percent accuracy is {}%".format(round(percent, 2)))

########################################################################
# This function is used to evaluate the car evaluation data set in hopes
# to determine if the car is a good buy or not. The output is averaging 
# a 90% accuracy rate. This is implemented with the KNN algorithm.
########################################################################
def car_evaluator():
    # For debugging
    print("calling car_evaluator()")

    carUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    carsDf = pd.read_csv(carUrl,names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classify'])

    # The new values I am going to replace
    new_values = {
    "buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1}, 
    "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1}, 
    "doors": {"2": 0, "3": 1, "4": 2, "5more": 3}, 
    "persons": {"2": 0, "4": 1, "more": 2}, 
    "lug_boot": {"small": 0, "med": 1, "big": 2}, 
    "safety": {"low": 0, "med": 1, "high": 2}, 
    "classify": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}

    carsDf.replace(new_values, inplace=True)

    # Convert to numPy
    df = carsDf.values

    # Train test and split this bad boi
    x_train, x_test, y_train, y_test = train_test_split(df[:,:-1], df[:,6], test_size=0.3, train_size=0.7)

    # Make and train my model  
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    # Predict output
    predicted = model.predict(x_test)

    calculate_display_percentage(predicted, y_test)
    
##########################################################################
# Function meant to test the autism data set to predict if someone is 
# autistic based on some categories that were previously defined. The
# average percentage was 25%. I believe there is something wrong with
# this algorithm. 
##########################################################################
def autism_evaluator():
    # For debugging
    print("calling autism_evaluator()")

    # Read the values in with no header and replace ? values
    autismDf = pd.read_csv('Autism-Adult-Data.csv', na_values=['?'], header=None)

    # Delete the second to last column about who filled out the information
    # There were too many missing values for me to try to fix
    autismDf.drop(autismDf.columns[[18, 19]], axis=1, inplace=True)

    # Label encoding with sklearn
    # This is good enough for me - Joe
    le = preprocessing.LabelEncoder()
    df_encoded = autismDf.apply(le.fit_transform)

    df_encoded.to_csv("cleanedAutism.csv")

    df = df_encoded.values

    x_train, x_test, y_train, y_test = train_test_split(df[:,:-1], df[:,17], test_size=0.3, train_size=0.7)

    # Make and train my model  
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    # Predict output
    predicted = model.predict(x_test)

    calculate_display_percentage(predicted, y_test)

#########################################################################
# Predicts the mpg from a dataset of car information, I am getting 10% 
# on this one for some reason
#########################################################################
def mpg_guesser():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

    df = pd.read_csv(url, header=None, delim_whitespace=True, na_values="?")

    # print("columns: {} rows: {}".format(len(df.columns), len(df.index)))

    df.drop(df.columns[[8]], axis=1, inplace=True)


    mode_column = mode(df.iloc[:,3])
    

    df = df.fillna(int(mode_column.mode))

    print(df.info())

    # Label encoding with sklearn
    # This is good enough for me - Joe
    le = preprocessing.LabelEncoder()
    df = df.apply(le.fit_transform)

    dfnum = df.values

    x_train, x_test, y_train, y_test = train_test_split(dfnum[:,1:8], dfnum[:,0], test_size=0.3, train_size=0.7)


    if np.isnan(x_train).any():
        print("NAAAAN")

    df.to_csv("lol.txt")

    print("x_train: {} y_train: {}".format(x_train, y_train))
    # Make and train my model  
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    # Predict output
    predicted = model.predict(x_test)

    calculate_display_percentage(predicted, y_test)



#########################################################################
# Where the methods get called
#########################################################################
# car_evaluator()

# autism_evaluator()

mpg_guesser()