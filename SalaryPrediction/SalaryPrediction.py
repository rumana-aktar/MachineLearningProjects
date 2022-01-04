# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 01/04/2022
#   
#   Problem: Predict the salary based on Years for experience using machine learning
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import os; 
clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# read the data
data = pd.read_csv("Salary.csv")

# x is the 'YearsExperience' column
x = data.drop('Salary', axis = 1); #print(x[0:10])

# y is the 'Salary' column
y = data['Salary']; #print(y.head())

# random_state = 42 means: test and train data will be picked randomly
# test_size = 0.2 means: 80% data will be used for training and 20% data will be used for testing

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

# # see the training and testing data
# print(xtrain); print(xtest); print(ytrain); print(ytest)

# use a linear model
L = LinearRegression()
L.fit(xtrain, ytrain)

# the prediction for the xtest;
y_pred = L.predict(xtest)


# see how well your model is
print(L.score(xtest, ytest))

#-------------- plot the result------------------

plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtrain, L.predict(xtrain), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(["model", "training data"])
#plt.show()
plt.savefig("trainingData_model.png")
plt.close()

plt.scatter(xtest, ytest, color = 'red')
plt.plot(xtrain, L.predict(xtrain), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(["model", "test data"])
#plt.show()
plt.savefig("testData_model.png")
plt.close()

plt.scatter(xtest, ytest, color = 'red')
plt.scatter(xtest, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(["test data", "prediction"])
#plt.show()
plt.ylim(4000, 140000)

plt.savefig("testData_predictionData.png")
plt.close()





