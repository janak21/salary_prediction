#!/usr/bin/python3

#Import all required libraries
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

#Loading dataset using pandas library
dataset = pd.read_csv("Salary_Data.csv")

#Check all columns (labels)
dataset.columns

#Print some entries
dataset.head()

#Make YearsExperience dependent variable
x = dataset['YearsExperience']
#Make Salary column as target attribute (independent variable)
y = dataset['Salary']

#Convert data into 2D array
x = x.values

x = x.reshape(-1, 1)

#Create Linear Regression model
model = LinearRegression()

#Feeding x and y to model
model = model.fit(x, y)

#Predicting salary for experience of 2 years
model.predict([[2.0]])

#Save model to local storage using joblib
joblib.dump(model, 'salaryModel.pkl')

#Load the saved model and predict
newmodel = joblib.load("salaryModel.pkl")

newmodel.predict([[2.0]])


