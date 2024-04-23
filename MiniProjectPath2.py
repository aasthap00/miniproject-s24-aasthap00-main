import pandas
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression



''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data


df = dataset_2[["Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Total"]]
# print(df.corr())

dataset_2['High Temp']  = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
dataset_2['Low Temp']  = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
dataset_2['Precipitation']  = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))

df1 = pandas.DataFrame(dataset_2[["High Temp", "Low Temp", "Precipitation", "Total"]])
print(df1.head())
print(df1.corr())

X = np.array(df1[["High Temp", "Low Temp", "Precipitation"]])
Y = np.array(df1[["Total"]])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


# MSE = mean_squared_error(y_test,y_pred)
# print(MSE)


poly = PolynomialFeatures(degree=3)

X_ = poly.fit_transform(X_train)
predict = poly.fit_transform(X_test)
clf = linear_model.LinearRegression()
clf.fit(X_,y_train)
y_pred = clf.predict(predict)

MSE = mean_squared_error(y_pred,y_test)
print(MSE)


# model = Pipeline([
#     ('polynomial_features', PolynomialFeatures(degree=3)),
#     ('linear_regression', LinearRegression())
# ])

# # Fit the model
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)

# MSE = mean_squared_error(y_pred,y_test)
# print(MSE)

