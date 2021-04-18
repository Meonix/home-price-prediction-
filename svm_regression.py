import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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
#regressor = SVR(kernel = 'rbf')
#regressor.fit(X, y)
#y_pred = regressor.predict(np.array(test_array).reshape(1, 16))


#2 using train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(np.array(X_test))
print('evaluating estimator performance: '+str(regressor.score(X_test, y_test)))
print(str(int(y_pred[0])) + '$')
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)




