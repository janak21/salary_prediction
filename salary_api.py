#!/usr/bin/python3
print("content-type: text/html")
print()

import joblib
import cgi

model = joblib.load("salaryModel.pkl")
y = cgi.FieldStorage()
get_value = y.getvalue("years")
exp = "{}.format(get_value)"
predict = model.predict([[int(get_value)]])
print(predict)
