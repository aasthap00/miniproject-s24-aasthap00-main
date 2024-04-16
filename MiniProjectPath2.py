import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


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

X = np.array(df1[["High Temp", "Low Temp", "Precipitation"]])
Y = np.array(df1[["Total"]])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


MSE = np.square(np.subtract(y_test,y_pred)).mean()
print(MSE)





