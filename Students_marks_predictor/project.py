# -*- coding: utf-8 -*-
"""project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16aAzO7HiqtrE_f5-9mK_8oiPQFXHHLYH

#Importing the required libraries
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

"""# Importing the dataset"""

df = pd.read_csv("student-mat.csv",sep = ";")
df1 = pd.read_csv("Getting data.csv")
df1

"""# Making data suitable for the model"""

X_test = df1.iloc[:, :3].values
y_test = df1.iloc[:, -1].values
X_train = df.iloc[:,-5:-2].values
y_train = df.iloc[:, -2].values
y_train  = y_train  + 5
X_train[:,-2] = X_train[:,-2] + 5

"""# Fitting Simple regression to data"""

regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""# Predicting the results"""

y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred,dtype = "int8")
y_pred = y_pred.clip(min=0,max = 25)
y_pred,y_test

"""# Checking the accuracy"""

y_pred
(1-np.sqrt(((y_pred - y_test) ** 2).mean())/15)*100
