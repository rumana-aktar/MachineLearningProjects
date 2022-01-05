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
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

#--------------------- Check the data information--------------------------------------------
data = pd.read_csv("student-mat.csv")
print(data.info())

#--------------------- check how each attribute affect G3--------------------------------------------
# # male performed better than female
# plt.bar(data['sex'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('sex')
# plt.show()

# # age has no relation with grade
# plt.bar(data['age'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('age'); plt.show()

# # famsize has positive relation with grade
# plt.bar(data['famsize'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('famsize'); plt.show()

# # Pstatus has relation with grade
# plt.bar(data['Pstatus'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('Pstatus'); plt.show()

# # Medu has positive relation with grade
# plt.bar(data['Medu'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('Medu'); plt.show()

# # Fedu has no relation with grade
# plt.bar(data['Fedu'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('Fedu'); plt.show()

# # Mjob no positive relation with grade
# plt.bar(data['Mjob'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('Mjob'); plt.show()

# # studytime might have positive relation with grade
# plt.bar(data['studytime'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('studytime'); plt.show()

# # failures has negative relation with grade
# plt.bar(data['failures'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('failures'); plt.show()

# # internet has positive relation with grade
# plt.bar(data['internet'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('internet'); plt.show()

# # romantic has negative relation with grade
# plt.bar(data['romantic'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('romantic'); plt.show()

# # Dalc has negative relation with grade
# plt.bar(data['Dalc'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('Dalc'); plt.show()

# # Walc no relation with grade
# plt.bar(data['Walc'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('Walc'); plt.show()

# # health has no relation with grade
# plt.bar(data['health'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('health'); plt.show()

# # absences has negative relation with grade
# plt.bar(data['absences'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('absences'); plt.show()

# # G1 has positive relation with grade
# plt.bar(data['G1'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('G1'); plt.show()

# # G2 has positive relation with grade
# plt.bar(data['G2'],data['G3'])
# plt.ylabel('G3'); plt.xlabel('G2'); plt.show()

#--------------------- so build the model using the dependecy --------------------------------------------
# Sex, famsize, pstatus, Medu, failure, internet, romantic, Dalc, absences, G1, G2 have relationship with G3

X=data[['G1','G2','Medu','Walc','studytime','failures','absences']]
y=data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.333, random_state = 101)


#--------------------- Using KNN model--------------------------------------------
print("--------------------- Using KNN model------------------------")
knn=KNeighborsRegressor(n_jobs=-1)
knn_neighbors={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}

classifier=GridSearchCV(knn,param_grid=knn_neighbors,cv=5,verbose=0).fit(X_train,y_train)
print(classifier.best_params_)

best_grid=classifier.best_estimator_
pred_1=best_grid.predict(X_test)

print("MAE: ", mean_absolute_error(y_test,pred_1))
print("MSE: ", mean_squared_error(y_test,pred_1))

#--------------------- Using Regression model--------------------------------------------
print("--------------------- Using Regression model------------------------")
L = LinearRegression()
L.fit(X_train, y_train)
pred_2 = L.predict(X_test)

print("MAE: ",mean_absolute_error(y_test,pred_2))
print("MSE: ",mean_squared_error(y_test,pred_2))
print("Accuracy: ", L.score(X_test, y_test))

#--------------------- Using Elastic Net--------------------------------------------
print("--------------------- Using Elastic Net------------------------")
elastic_net=ElasticNet(alpha=0.01,l1_ratio=0.5)
model_3=elastic_net.fit(X_train,y_train)
pred_3=model_3.predict(X_test)
print("MAE: ",mean_absolute_error(y_test,pred_3))
print("MSE: ",mean_squared_error(y_test,pred_3))
accuracy = elastic_net.score(X_test, y_test)
print('Accuracy: ',accuracy)


