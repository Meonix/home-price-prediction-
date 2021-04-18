import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
data = pd.read_csv("new_dataset.csv")
data.head()
data = data.drop(columns='id')
data = data.drop(columns='date')
data.head()

y = data.loc[:, 'price']
X = data.drop(columns='price')
with open('input.txt') as my_file:
    test_array = my_file.readlines()
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)

regressor = MLPRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict(np.array(test_array).reshape(1, 16))
print(str(int(y_pred[0])) + '$')