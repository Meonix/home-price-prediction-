import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
data = pd.read_csv("new_dataset.csv")
data.head()
data = data.drop(columns='id')
data = data.drop(columns='date')
data.head()

y = data.loc[:, 'price']
X = data.drop(columns='price')

#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

y_pred = regressor.predict(np.array([3,2.56,1700,4252,3,0,0,4,6,1722,800,1922,0,98114,47.2323,-122.243]).reshape(1, 16))
print(str(int(y_pred[0])) + '$')