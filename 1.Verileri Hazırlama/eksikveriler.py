# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:15:48 2019

@author: Erdem
"""


#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#kod bolumu
#verileri yükleme
#veriler = pd.read_csv('veriler.csv')
veriler = pd.read_csv('eksikveriler.csv')


print(veriler)

#veri önişleme
boy=veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

#eksik veriler

from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0,)

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)