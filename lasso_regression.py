# evaluate an lasso regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
# load the dataset
data = pd.read_csv("new_dataset.csv")
data.head()
data = data.drop(columns='id')
data = data.drop(columns='date')
data.head()

y = data.loc[:, 'price']
X = data.drop(columns='price')
print('________________________________')
print('calculating.....')
# 1 using input data
#with open('input.txt') as my_file:
#    test_array = my_file.readlines()
#model = Lasso()
#model.fit(X, y)
#yhat = model.predict(np.array(test_array).reshape(1, 16))
# summarize prediction
#print('Predicted: %.3f' % yhat)

#2 using train data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
model = Lasso()
model.fit(X_train, y_train)
yhat = model.predict(np.array(X_test))
#Đánh giá mô hình
print('evaluating estimator performance: '+str(model.score(X_test, y_test)))


print(yhat[0])
# define model evaluation method
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
#print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))