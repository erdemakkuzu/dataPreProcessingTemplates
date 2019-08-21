#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#Numpy Array dönüşümü
X=x.values
Y=y.values



#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#pollynomial regression
#doğrusal olmayan non linear model oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)



#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4. dereceden polinom
poly_reg3=PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)


lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


#görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X))
plt.show()
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli= sc2.fit_transform(Y)

    
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
deger=svr_reg.predict(11)

#Decision tree regression
from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X),color='blue')
plt.show()
treedeger=r_dt.predict(6.6)

#Random forrest Regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)
randomforrest=rf_reg.predict(6.5)
plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X), color='blue')
plt.show()


rsquared=r2_score(Y,rf_reg.predict(X))
print("Random forrest r2 değeri")
print(rsquared)











