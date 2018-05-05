# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:39:57 2018

EE511 Project #5, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10
Tested in Python 3.5.4 :: Anaconda custom (64-bit), Windows 10

Question 1.
Markov Chain Monte Carlo : Metropolis-Hastings algorithm

"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, cauchy, ks_2samp

def f(x):
    # Define the input mixture distribution
    a1 = 1; b1 = 8; a2 = 9; b2 = 1    
    y = (0.6 * beta.pdf(x, a1, b1)) + (0.4 * beta.pdf(x, a2, b2 ))
    return y

def q1(xt):
    # Define a proposal pdf q(x)
    # This function returns a randomly generated sample 
    # from the defined Cauchy distribution
    y = cauchy.rvs(loc = xt, scale=0.2, size=1)
    return y

def q2(xt):
    # Define a proposal pdf q(x)
    # This function returns a randomly generated sample 
    # from the defined Normal distribution
    y = norm.rvs(loc = xt, scale = 0.5)
    return y

def plots(sample, title):
    y1 = f(sample)
    plt.figure(2)
    plt.plot(sample, y1, 'bo'); plt.xlabel('x'); plt.ylabel('y=f(x)')
    plt.title('Samples drawn from the mixture using MCMC sampling ('+title+')')
    
    plt.figure(3)
    plt.plot(sample, y1); plt.xlabel('x'); plt.ylabel('y=f(x)')
    plt.title('Sample path for samples drawn')
    
def fit_test(samples):
    y1 = f(samples)
    rvs1 = norm.pdf(samples)
    d, p_value = ks_2samp(y1, rvs1)
    print('The p-value give by the K-S test:', p_value)
    
    
# Try to avoid writing this line the program as this shuts 
# down few important too
warnings.filterwarnings('ignore')

x = np.linspace(-0.1, 1.1, num = 1000)
y = f(x)
plt.figure(1)
plt.plot(x, y) ; plt.xlabel('x') ; plt.ylabel('y = f(x)')
plt.title('Graph of the mixture distribution')

while True:
    x0 = np.random.rand()
    if f(x0) != 0:
        break

n_samples = 50000
sample = np.empty([n_samples])
sample[0] = x0
n_count = 0
while True:
    if (n_count == 0):
        y = q2(sample[0])
    else:
        y = q2(sample[n_count - 1])
    
    if (y>=0 and y<=1):
        ratio = f(y) / f(sample[n_count - 1])
        # Acceptance probability
        alpha = min(1, ratio)
        
        u = np.random.rand()
        if (u <= alpha):
            n_count = n_count + 1
            #print(n_count)
            sample[n_count] = y
    
    if (n_count == n_samples - 1):
        break    
sample1 = sample[45000:]    
plots(sample1, 'Cauchy distribution')
#plots(sample1, 'Normal distribution')
fit_test(sample1)