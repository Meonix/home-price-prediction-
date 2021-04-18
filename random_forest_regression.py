import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

data = pd.read_csv("new_dataset.csv")
data.head()
data = data.drop(columns='id')
data = data.drop(columns='date')
data.head()

y = data.loc[:, 'price']
X = data.drop(columns='price')
# 1 using input data
#with open('input.txt') as my_file:
#    test_array = my_file.readlines()
#regressor = RandomForestRegressor()
#regressor.fit(X, y)
#y_pred = regressor.predict(np.array(test_array).reshape(1, 16))
print('________________________________')
print('calculating.....')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#sc = StandardScaler()   
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#regressor = RandomForestRegressor(n_estimators=20, random_state=0)
#2 using train data
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
#Đánh giá mô hình
y_pred = regressor.predict(np.array(X_test))
print('evaluating estimator performance: '+str(regressor.score(X_test, y_test)))
print(y_pred[0])

