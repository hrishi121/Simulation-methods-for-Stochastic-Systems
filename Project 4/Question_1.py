# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 12:07:44 2018

EE511 Project #4, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1.

"""

import numpy as np
import matplotlib.pyplot as plt

################################ Part (a) #####################################

samples = 100
pi_pred = np.empty([50])
for i in range(0, 50):
    x = np.random.rand(samples, 2)
    count = ( (x[:,0]**2 + x[:,1]**2 ) < 1).sum()
    pi_pred[i] = (count/samples) * 4
    
print("Average Estimated value of pi: ", np.mean(pi_pred))

plt.figure(1)
plt.hist(pi_pred, histtype = 'bar', facecolor='green', alpha=0.75)
plt.xlabel('Estimated value of Pi')
plt.ylabel('# of times')
plt.title('Estimating the value of π using Monte Carlo simulation')
plt.grid(True)
plt.show()

################################ Part (b) #####################################

n = np.logspace(2, 4)   # Range of uniform samples selected
var_n = np.empty([50])  # Sample variance for different values of n
k = 0
for j in n:
    samples = int(np.floor(j))
    pi_pred = np.empty([50])
    for i in range(0, 50):
        x = np.random.rand(samples, 2)
        count = ( (x[:,0]**2 + x[:,1]**2 ) < 1).sum()
        pi_pred[i] = (count/samples) * 4
    var_n[k] = np.var(pi_pred)
    k = k + 1
    
t = np.floor(n)
plt.figure(2)
plt.plot(t, var_n, '+g-', alpha=0.75)
plt.xlabel('# of uniform samples selected (n)')
plt.ylabel('Sample variance')
plt.title('Graph of Sample variance of π - estimates for different values of n')
plt.grid(True)
plt.show()



