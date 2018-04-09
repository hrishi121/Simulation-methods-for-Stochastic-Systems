# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:30:50 2018

EE511 Project #4, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2c
"""

import numpy as np
from math import cos, pi
from scipy.integrate import dblquad

############################## Actual Integration #############################

area = dblquad(lambda x, y: 20 + x**2 + y**2 - 10*(cos(2*pi*x) + cos(2*pi*y)),
               -5, 5, lambda x: -5, lambda x: 5)

print('Actual value of the definite integral: {0:.3f}'.format(area[0]))

############################ Monte Carlo Estimation ###########################

samples = 1000
y1 = np.zeros([50])
rv = np.empty([samples, 2])

for i in range(0, 50):
    rv = 10 * np.random.random_sample([samples, 2]) - 5
    for w in range(0, samples):
        temp = 20 + rv[w, 0]**2 + rv[w, 1]**2 - 10*(cos(2*pi*rv[w, 0]) + cos(2*pi*rv[w, 0]))
        y1[i] = y1[i] + temp
        #temp2 = np.divide(temp1, pf)
    y1[i] = (y1[i] / samples) * 100
final1 = np.mean(y1)
print('Value of the definite integral using Monte Carlo estimation: {0:.3f}'.format(final1))

