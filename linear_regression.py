import numpy as np
import pandas as pd

df = pd.read_csv("home_price_prediction_dataset.csv")
df.head()

#tính mean của từng cột trong dataset
meanPrice = sum(df['price']) / float(len(df['price']))
meanBedrooms = sum(df['bedrooms']) / float(len(df['bedrooms']))
meanBathrooms = sum(df['bathrooms']) / float(len(df['bathrooms']))
meanSqftLiving = sum(df['sqft_living']) / float(len(df['sqft_living']))
meanSqftLot = sum(df['sqft_lot']) / float(len(df['sqft_lot']))
meanFloors = sum(df['floors']) / float(len(df['floors']))
meanWaterfront = sum(df['waterfront']) / float(len(df['waterfront']))
meanView = sum(df['view']) / float(len(df['view']))
meanCondition = sum(df['condition']) / float(len(df['condition']))
meanGrade = sum(df['grade']) / float(len(df['grade']))
meanSqftAbove = sum(df['sqft_above']) / float(len(df['sqft_above']))
meanSqftBasement = sum(df['sqft_basement']) / float(len(df['sqft_basement']))
meanYearBuilt = sum(df['yr_built']) / float(len(df['yr_built']))
meanYearRenovated = sum(df['yr_renovated']) / float(len(df['yr_renovated']))
meanZipcode = sum(df['zipcode']) / float(len(df['zipcode']))
meanLat = sum(df['lat']) / float(len(df['lat']))
meanLong = sum(df['long']) / float(len(df['long']))


def variance(values, mean):# **2 nghĩa là ^2
    return sum([(val-mean)**2 for val in values])
    
def covariance(column1, meanColumn1, column2 , meanColumn2): #thể hiện mối quan hệ giữa hai biết ( đồng biến , nghịch biến)
    covariance = 0.0
    for r in range(len(column1)):
        covariance = covariance + (column1[r] - meanColumn1) * (column2[r] - meanColumn2)
    return covariance
    
def linearFitment(df1,df2,mean1,mean2,params): #df1 là values của cột được truyền vào , mean1 là mean được truyền vào , params là thông số người dùng nhập
    variance1 = variance(df1, mean1) 
    variance2  =  variance(df2, mean2)

    covariance_1_2 = covariance(df1,mean1,df2,mean2)

    m = covariance_1_2/variance1
    c = mean2 - m * mean1
    return m * params + c
print(linearFitment(df['price'],df['bedrooms'],meanPrice,meanBedrooms,1230000))

