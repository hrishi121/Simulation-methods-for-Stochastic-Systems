# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:09:46 2018

EE511 Project #2, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2a.

"""
import numpy as np
import random as rand
from math import *
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.stats import chisquare

def exponentialRNG(numSamples, avgTime):
    # This function creates random numbers with exponential
    # distribution using inverse transform method
    # numSamples: # of samples to be generated
    # avgTime: average waiting time(lambda parameter)
    
    expSample = np.empty([numSamples])

    for i in range(0, numSamples):
        x = rand.random()
        expSample[i] = (-1) *avgTime * log(1 - x)

    return expSample

def main():
    avgTime = 0.2          # avgTime: average waiting time(lambda parameter)
    numSamples = 1000      # numSamples: # of samples to be generated
    numBins = 10           # nunBins: This is the # of bins for the 
                           
    bins = [0, 0.044, 0.1021, 0.1832, 0.3219, 1 ]
                           
    expSample = exponentialRNG(numSamples, avgTime)
    
    e = expon.rvs(size = numSamples, scale = avgTime)
    # Generates random numbers from an exponential continuous random variable.
  
    
    plt.figure(1)    
    plt.hist(expSample, numBins, color = 'blue', hatch = '/')
    plt.title("Histogram of 1000 samples of exponential random variables(using Inverse transform method)", alpha = 0.75) 
    plt.xlabel('Waiting Time')
    plt.ylabel('# Samples')
    plt.grid(True)
    plt.savefig("Observed_Histogram.png")
    plt.show()
    
    plt.figure(2)  
    plt.hist(e, numBins, color = 'red', hatch = '/')
    plt.title("Histogram of 1000 samples of exponential random variables(using inbuilt function) ", alpha = 0.75)
    plt.xlabel('Waiting Time')
    plt.ylabel('# Samples')
    plt.grid(True)
    plt.savefig("Expected_Histogram.png")
    plt.show()
    
    (observedValues, bins, patches) = plt.hist(expSample, bins, color = 'blue', hatch = '/')
    print("Observed Values: ",observedValues,"\n")
    
    (expectedValues, bins, patches) = plt.hist(e, bins, color = 'red', hatch = '/')
    print("Expexted Values: ",expectedValues,"\n")
    
    
    (chi) = chisquare(f_obs = observedValues, f_exp = expectedValues)
    print(chi)
    
if __name__ == "__main__":
    main()



