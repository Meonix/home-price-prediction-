import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("new_dataset.csv")
data.head()
data = pd.concat([pd.Series(1.0, index=data.index, name='00'), data], axis=1)
data.head()
data = data.drop(columns='id')
data = data.drop(columns='date')

#data = (data - data.mean())/data.std()
data.head()
with open('input.txt') as my_file:
    test_array = my_file.readlines()
Y = data.loc[:, 'price']
X = data.drop(columns='price')

w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X),X)),np.transpose(X)),Y)

print(int(w[0]+w[1]*float(test_array[0])+w[2]*float(test_array[1])+w[3]*float(test_array[2])+w[4]*float(test_array[3])+w[5]*float(test_array[4])+w[6]*float(test_array[5])+
w[7]*float(test_array[6])+w[8]*float(test_array[7])+w[9]*float(test_array[8])+w[10]*float(test_array[9])+w[11]*float(test_array[10])+w[12]*float(test_array[11])+w[13]*float(test_array[12])+w[14]*float(test_array[13])+w[15]*float(test_array[14])+w[16]*float(test_array[15])))
