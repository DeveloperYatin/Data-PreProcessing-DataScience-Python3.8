# -*- coding: utf-8 -*-
# Yatin Batra | yatinbatra31@gmail.com | Aka Developer Yatin
"""
Created on Fri Jul 17 14:23:45 2020

@author: Developer Yatin
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#dataset = data.drop_duplicates(subset="YearsExperience",keep='first',inplace=False) #Removing duplicate values
X = dataset.iloc[:,:-1].values # All rows, all columns except last
y = dataset.iloc[:, 1].values # All rows, first column (index starts at 0) column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Line fitting
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
y_pred = regression.predict(X_train)

# Plotting the plot on training data
plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,regression.predict(X_train),color="red")
plt.title("Training Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Plotting the plot on testing data
plt.scatter(X_test,y_test,color="blue")
plt.plot(X_train,regression.predict(X_train),color="red")
plt.title("Test Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Predicting random values 
a = 1.1
pred_b = np.array(a)
pred_b = pred_b.reshape(1, -1)
b = regression.predict(pred_b)
print(round(float(b),2))


#Saving model joblib package file

#import joblib
#joblib.dump(regression, "linear_regression_model.pkl")

#Testing joblib package file

#a = 1.1
#pred_b = [a]
#pred_b_arr = np.array(pred_b)
#pred_b_arr = pred_b_arr.reshape(1, -1)
#linear_reg = open("linear_regression_model.pkl","rb")
#ml_model = joblib.load(linear_reg)
#model_prediction = ml_model.predict(pred_b_arr)
#print(round(float(model_prediction), 2))


#Testing accuracy of model

accuracy = regression.score(X_test,y_test)
print(accuracy*100,'%')


