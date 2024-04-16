import pandas

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data


def trafficModel(bridge1, bridge2, bridge3):
    x = dataset_2(f"{bridge1} Bridge", f"{bridge2} Bridge", f"{bridge3} Bridge").values
    y = dataset_2("Total").values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = linearRegression()
    model.fit(x_train, y_train)