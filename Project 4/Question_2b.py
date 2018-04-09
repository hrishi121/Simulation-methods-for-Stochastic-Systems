# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:43:55 2018

EE511 Project #4, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2b
"""

import numpy as np
from math import exp, pi
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

############################## Actual Integration #############################

area = dblquad(lambda x, y: exp(-1*(x**4 + y**4)), -pi, pi,
               lambda x: -pi, lambda x: pi)

print('Actual value of the definite integral: {0:.3f}'.format(area[0]))

############################ Monte Carlo Estimation ###########################

samples = 1000
y1 = np.zeros([50])
y2 = np.zeros([50])
y3 = np.zeros([50])
rv1 = np.empty([samples, 2])
rv2 = np.empty([samples, 2])
count = 0

for i in range(0, 50):
    rv1 = 2*pi * np.random.random_sample([1000, 2]) - pi
    for w in range(0, samples):
        temp = exp(-1 * (pow(rv1[w, 0], 4) + pow(rv1[w, 1], 4)))
        y1[i] = y1[i] + temp
        #temp2 = np.divide(temp1, pf)
    y1[i] = (y1[i] / samples) * (4 * pi *pi)
final1 = np.mean(y1)
print('\n Value of the definite using Monte Carlo estimation:')
print('1. Without variance reduction : {0:.3f} ; Sample Variance: {1:.3f}'.format(final1, np.var(y1)))

############################ Importance Sampling ##############################

for i in range(0, 50):
    #Generate random samples from the proposal PDF
    while True:
        if count > samples - 1:
            break
        t1 = mvn.rvs(mean = [0,0], cov = [[1, 0], [0, 1]])
        if (t1[0] <= pi and t1[1] <=pi and t1[0] >= -pi and t1[1] >= -pi):
            rv2[count, 0] = t1[0]
            rv2[count, 1] = t1[1]
            count = count + 1
        else:
            count = count + 1
    
    pf = mvn.pdf(rv2, mean = [0,0], cov = [[1, 0], [0, 1]])
    temp1 = np.empty([samples])
    for w in range(0, samples):
        temp1[w] = exp(-1 * (pow(rv2[w, 0], 4) + pow(rv2[w, 1], 4)))
    temp2 = np.divide(temp1, pf)
    y2[i] = (np.sum(temp2) / samples)
    
final2 = np.mean(y2)
print('2. Using Importance Sampling: {0:.3f} ; Sample Variance: {1:.3f}'.format(final2, np.var(y2)))


############################ Stratified sampling ###########################


for i in range(0, 50):
    rv1 = 2*1.73 * np.random.random_sample([1000, 2]) - 1.73
    for w in range(0, samples):
        temp = exp(-1 * (pow(rv1[w, 0], 4) + pow(rv1[w, 1], 4)))
        y3[i] = y3[i] + temp
        #temp2 = np.divide(temp1, pf)
    y3[i] = (y3[i] / samples) * (4 * 1.73 * 1.73)
final3 = np.mean(y3)
print('3. Using stratified Sampling: {0:.3f} ; Sample Variance: {1:.3f}'.format(final3, np.var(y3)))

############################# Plt the function ################################

t = np.linspace(-pi, pi)
x2, y2 = np.meshgrid(t, t)
rv3 = np.empty([50, 50])
for i in range(0, 50):
    for j in range(0, 50):
        temp = pow(x2[i, j], 4) + pow(y2[i,j], 4)
        rv3[i, j] = exp(-1*temp)

plt.figure(2)
plt.contourf(x2, y2, rv3)
plt.colorbar()
plt.xticks(); plt.yticks()
plt.xlabel('x') ; plt.xlabel('y')
plt.title('Contour plot of the 2nd function')
plt.show()

plt.figure(3)
ax = plt.axes(projection='3d')
ax.plot_surface(x2, y2, rv3,  cmap='viridis', edgecolor='none')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z');
ax.set_title('Graph of the 2nd function')
plt.show()