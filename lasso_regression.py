# evaluate an lasso regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
# load the dataset
data = pd.read_csv("new_dataset.csv")
data.head()
data = data.drop(columns='id')
data = data.drop(columns='date')
data.head()

y = data.loc[:, 'price']
X = data.drop(columns='price')

with open('input.txt') as my_file:
    test_array = my_file.readlines()
print('________________________________')
print('calculating.....')
# define model
model = Lasso()
# fit model
model.fit(X, y)
# define new data
#row = [3,2.56,1700,4252,3,0,0,4,6,1722,800,1922,0,98114,47.2323,-122.243]
# make a prediction
yhat = model.predict(np.array(test_array).reshape(1, 16))
# summarize prediction
print('Predicted: %.3f' % yhat)

#Đánh giá mô hình
# define model evaluation method
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))