# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:15:48 2019

@author: Erdem
"""


#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri önişleme


#Veri Yükleme
veriler = pd.read_csv('satislar.csv')


print(veriler)


aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)



#verilerin eğitim ve test için bölünmesi
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train , y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)









