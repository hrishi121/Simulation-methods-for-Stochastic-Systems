# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:06:51 2018

EE511 Project #5, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10
Tested in Python 3.5.4 :: Anaconda custom (64-bit), Windows 10

Question 2.
Markov Chain Monte Carlo : Simulated Annealing

"""

import numpy as np
import math
import sys
from math import exp, log
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import beta, norm, cauchy
from scipy.stats import multivariate_normal as mvn


def f(x1, x2):
    '''
    Define the input Schwefel function
    
    '''
    y = 418.9829 * 2 - (    x1*np.sin(np.sqrt(abs(x1)))
                          + x2*np.sin(np.sqrt(abs(x2)))     )
    return y

def q(x1, x2):
    '''
    Define a proposal pdf q(x)
    
    '''

    q1 = np.empty([2])
    #q1[0] = norm.rvs(loc = x1, scale = 1)
    #q1[1] = norm.rvs(loc = x2, scale = 1)
    
    q1[0] = cauchy.rvs(loc = x1, scale=0.2, size=1)
    q1[1] = cauchy.rvs(loc = x2, scale=0.2, size=1)
    return q1

def t1(i):
    # Cooling function : lograthmic
    t = 500/(1 + log(i + 1))
    return t


def t2(i):
    # Cooling function : exponential
    t = 0.8*i
    return t

def t3(i):
    # Cooling function : linear
    t = 500/(1 + 2*i)
    return t

def t4(i):
    # Cooling function : quadratic
    t = 500/(1 + 2*i*i)
    return t


def updt(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rRuns: [{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    

itf = lambda length: math.ceil(1.2*length)


x1_start = 0; x2_start = 0
length = 100

best_sample = np.array([x1_start, x2_start])
best_run = np.empty([1,1])
best_sample_runs = np.array([x1_start, x2_start])
accept_data = list(); reject_data = list()


for j in range(0, 100):
    
    updt(100, j + 1)
    
    t_curr = 20
    samples = np.array([[x1_start, x2_start]])
    accept = 0; reject = 0; count = 0
    
    for i in range(0, 1000):
        
        if (i%50 == 0):
            t_curr = t1(i)
               
        while True:
            y = q(samples[count, 0], samples[count, 0])
            if(y[0]<=500 and y[0]>=-500 and y[1]<=500 and y[1]>=-500):
                break
            
        f_prev = f(samples[count, 0], samples[count, 0])
        f_curr = f(y[0], y[1])
        
        ratio = np.exp((f_prev - f_curr)/t_curr)
        alpha = min(1, ratio)
        u = np.random.uniform()
        
        if((f_curr < f_prev) or (u <= alpha)):
            count += 1; accept +=1
            r = np.array([[y[0], y[1]]])
            samples = np.vstack((samples, r))
            if(f_curr < f_prev):
                best_sample = y
        
        else:
            reject += 1
            
    accept_data.append(accept)
    reject_data.append(reject)
    
    r1 = f(best_sample_runs[0], best_sample_runs[1])
    r2 = f(best_sample[0], best_sample[1])

    if(r1 > r2):
        best_sample_runs = best_sample
        best_run = samples

count = len(best_run)
fy = np.zeros([count])
for i in range(0, count):
    fy[i] = f(best_run[i, 0], best_run[i, 0])
plt.figure(2)
plt.hist(fy); plt.xlabel('Values of Minima'); plt.ylabel('#')
plt.title('Histogram of minima')
 

x1 = np.linspace(-500, 500, num = 501)
x2 = np.linspace(-500, 500, num = 501)

xx1, xx2 = np.meshgrid(x1, x2, indexing ='xy')
y = f(xx1, xx2)
plt.figure(3)
plt.contourf(xx1, xx2, y, cmap = 'viridis', alpha = 0.7)
plt.plot(best_run[:,0], best_run[:,0], 'r')
plt.colorbar()
plt.title('Contour plot of the Scwefel function')
plt.xlabel('x1'); plt.ylabel('x2')

     
        
        
        

        

