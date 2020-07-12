# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:52:58 2020

@author: Yatin
"""

# Importing libraries

import numpy as np
#import matplotlib.pyplot as mtp
import pandas as pd


#importing datasets

data_set = pd .read_csv('Data.csv')


#Extracting independent variable

x = data_set.iloc[:,:-1].values

#Extracting dependent variable

y = data_set.iloc[:,3].values

#Handling missing data (Replacing missing data with the mean value)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

#Fitting imputer object to the independent variables x.
imputerimputer = imputer.fit(x[:,1:3])

#Replacing missing data with the calculated mean value 
x[:,1:3] = imputer.transform(x[:,1:3])


#Catgorical data
# for Country variable 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lable_encoder_x = LabelEncoder()

x[:,0] = lable_encoder_x.fit_transform(x[:,0])

#Encoding for dummy variable

ct = ColumnTransformer(transformers=[("Country",OneHotEncoder(),[0])],remainder = 'passthrough')

x = np.array(ct.fit_transform(x),dtype = np.float)

#onehot_encoder = OneHotEncoder(categories=[0])     #For python previous versions

#x = onehot_encoder.fit_transform(x).toarray()

#Encoding for purchased variable 

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#Splitting the dataset into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling of datasets

from sklearn.preprocessing import StandardScaler

st_x = StandardScaler()

x_train = st_x.fit_transform(x_train)

x_test = st_x.transform(x_test)
