# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:10:24 2018

code reference from http://scikit-learn.org/stable/modules/multiclass.html

@author: wangxinji
"""

import pandas as pd
import os
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
os.chdir('C:/Users/Hasee/Desktop')
#code reference from http://scikit-learn.org/stable/modules/multiclass.html

def multiacc(dataset,predict):
    match =0
    for i in range(len(predict)):
        if float(dataset[i][-1]) == predict[i]:
            match = match +1
    acc = match/len(predict)
    return acc 

wine = pd.read_csv('wine.csv',header = None)
wine = wine.values
x = wine[:,:-2]
y = wine[:,-1]
result = OneVsOneClassifier(LinearSVC(random_state = 0)).fit(x,y).predict(x)

multiacc(wine,result)
