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
veriler = pd.read_csv('veriler.csv')


print(veriler)

#veri önişleme
boy=veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)