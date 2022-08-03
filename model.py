# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('ds_salaries.csv')
Y = dataset['salary_in_usd']
X = dataset.drop(['salary_in_usd'], axis=1)


def convert_to_int(string):
    dict = {'EN':1, 'EX':2, 'MI':3, 'SE':4}
    return dict[string]

X['experience_level'] = X['experience_level'].apply(lambda x : convert_to_int(x))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()



#Fitting model with trainig data
regressor.fit(X, Y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5, 2]]))