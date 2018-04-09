# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:35:14 2018

EE511 Project #4, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2a

"""


import numpy as np
from math import exp, pi, sinh, log
from scipy.integrate import quad
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

############################## Actual Integration #############################

area = quad(lambda x: (1 / (1 + sinh(2*x) * log(x)) ), 0.8, 3)

print('Actual value of the definite integral: {0:.3f}'.format(area[0]))

############################ Monte Carlo Estimation ###########################

samples = 1000
y1 = np.zeros([50])
y2 = np.zeros([50])
y3 = np.zeros([50])
count = 0

for i in range(0, 50):
    rv1 = 2.2 * np.random.random_sample([1000]) + 0.8
    for w in range(0, samples):
        temp = pow(1 + np.multiply(np.sinh(2*rv1[w]), np.log(rv1[w]) ), -1)
        y1[i] = y1[i] + temp
        #temp2 = np.divide(temp1, pf)
    y1[i] = (y1[i] / samples) * (2.2)
final1 = np.mean(y1)
print('\n Value of the definite using Monte Carlo estimation:')
print('1. Without variance reduction is: {0:.3f} ; Sample Variance: {1:.3f}'.format(final1, np.var(y1)))

############################ Importance Sampling ##############################

samples = 1000

# parameters for the proposal PDF
lower, upper = 0.8, 3
mu, sigma = 0.45, 0.5

for i in range(0, 50):
    #Generate random samples from the proposal PDF
    x = truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma,
            loc = mu, scale = sigma, size = samples)
    
    pf = truncnorm.pdf(
            x, (lower - mu) / sigma, (upper - mu) / sigma,
            loc = mu, scale = sigma)
    
    temp1 = pow(1 + np.multiply(np.sinh(2*x), np.log(x) ), -1)
    temp2 = np.divide(temp1, pf)
    y2[i] = np.sum(temp2) / samples
    
final2 = np.mean(y2)
print('2. Using Importance Sampling: {0:.3f} ; Sample Variance: {1:.3f} '.format(final2, np.var(y2)))

############################ Stratified Sampling ##############################


for i in range(0, 50):
    rv1 = 1.2 * np.random.random_sample([550]) + 0.8
    rv2 = np.random.random_sample([450]) + 2
    for w in range(0, 550):
        temp1 = pow(1 + np.multiply(np.sinh(2*rv1[w]), np.log(rv1[w]) ), -1)
        y3[i] = y3[i] + temp1
    for w in range(0, 450):
        temp2 = pow(1 + np.multiply(np.sinh(2*rv2[w]), np.log(rv2[w]) ), -1)
        y3[i] = y3[i] + temp2
    y3[i] = (y3[i] / samples) * (2.2)
final3 = np.mean(y3)
print('3. Using Stratified Sampling: {0:.3f} ; Sample Variance: {1:.3f}'.format(final3, np.var(y3)))

############################# Plt the function ################################

x1 = np.linspace(0.1, 3)
x2 = np.linspace(0.8, 3)
rv1 = pow(1 + np.multiply(np.sinh(2*x1), np.log(x1) ), -1)
rv2 = pow(1 + np.multiply(np.sinh(2*x2), np.log(x2) ), -1)

plt.figure(1)
plt.plot(x1, rv1, x2, rv2)
plt.legend(('Extended function', 'Actual Function'))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of first function')
plt.grid(True)
plt.show()