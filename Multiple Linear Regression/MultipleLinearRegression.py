# Yatin Batra | yatinbatra31@gmail.com | Aka Developer Yatin
# Multiple Linear Regression on the '50_Startups' Dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])

ct = ColumnTransformer(transformers=[("Country",OneHotEncoder(),[3])],remainder = 'passthrough')

X = np.array(ct.fit_transform(X),dtype = np.float)

#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Avoiding the - dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

# Fitting the line
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#Predicting the test set result
y_pred = regressor.predict(X_test)



#Score {Coefficient of determination} of train model

print('Train Score: ', regressor.score(X_train,y_train))
print('Test Score: ', regressor.score(X_test,y_test))


