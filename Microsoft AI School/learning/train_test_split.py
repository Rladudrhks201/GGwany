import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv'
df = pd.read_csv(url)

columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'price']
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
df = df.loc[:, columns]
# print(df.head(3))

X = df.loc[:, features]
y = df.loc[:, ['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
