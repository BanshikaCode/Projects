import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


dataset = pd.read_csv(r"C:\Users\Bansh\OneDrive\Desktop\Project_github\CAR DETAILS FROM CAR DEKHO.csv")
print(dataset.info())
print(dataset.isnull().sum())

# chechking the distributionof categorical data
print(dataset['fuel'].value_counts())
print(dataset['seller_type'].value_counts())
print(dataset.transmission.value_counts())

# encoding fuel_type column
dataset.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True)
# encoding "seller_type "
dataset.replace({'seller_type':{'Individual':0,'Dealer':1,'Trustmark Dealer':2}},inplace=True)
# encoding "Transmission" 
dataset.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)

print(dataset.head())

X=dataset.drop(['name','selling_price','owner'],axis=1) #for column axis=1,row its 0
y=dataset['selling_price']

print(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2)

LR=LinearRegression()
LR.fit(X_train,y_train)
y_train_predict=LR.predict(X_train)

# R squared error
error_score=metrics.r2_score(y_train,y_train_predict)
print(error_score)


plt.scatter(y_train,y_train_predict)
# they are very close so goood

y_test_predict=LR.predict(X_test)
error_test_score=metrics.r2_score(y_test,y_test_predict)
print(error_test_score)

plt.scatter(y_test,y_test_predict)