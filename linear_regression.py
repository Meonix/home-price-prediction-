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

Y = data.loc[:, 'price']
X = data.drop(columns='price')

w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X),X)),np.transpose(X)),Y)
print("Enter bedrooms:")
bedrooms = input()
print("Enter bathrooms:")
bathrooms = input()
print("Enter sqft Living:")
sqftLiving = input()
print("Enter sqft Lot:")
sqftLot = input()
print("Enter floors:")
floors = input()
print("Enter waterfront:")
waterfront = input()
print("Enter number of views:")
views = input()
print("Enter condition:")
condition = input()
print("Enter grade:")
grade = input()
print("Enter sqft Above:")
sqftAbove = input()
print("Enter sqft Basement:")
sqftBasement = input()
print("Enter year Built:")
yrBuilt = input()
print("Enter year Renovated:")
yearRenovated = input()
print("Enter zipcode:")
zipcode = input()
print("Enter latitude:")
lat = input()
print("Enter longitude :")
long = input()
print(int(w[0]+w[1]*float(bedrooms)+w[2]*float(bathrooms)+w[3]*float(sqftLiving)+w[4]*float(sqftLot)+w[5]*float(floors)+w[6]*float(waterfront)+
w[7]*float(views)+w[8]*float(condition)+w[9]*float(grade)+w[10]*float(sqftAbove)+w[11]*float(sqftBasement)+w[12]*float(yrBuilt)+w[13]*float(yearRenovated)+w[14]*float(zipcode)+w[15]*float(lat)+w[16]*float(long)))