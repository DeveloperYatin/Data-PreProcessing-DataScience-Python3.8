# Yatin Batra | yatinbatra31@gmail.com | Aka Developer Yatin
# Multiple Linear Regression on the '50_Startups' Dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups - Optimal.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

# Fitting the MLR model to training the set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(np.array(X_train).reshape(-1,1),y_train)


#Predicting the test set result
y_pred = regressor.predict(X_test)



#Score {Coefficient of determination} of train model

print('Train Score: ', regressor.score(X_train,y_train))
print('Test Score: ', regressor.score(X_test,y_test))


