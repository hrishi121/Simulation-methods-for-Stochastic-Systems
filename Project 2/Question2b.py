# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:36:33 2018

EE511 Project #2, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2b.

"""


import numpy as np
import random as rand
from math import *
import matplotlib.pyplot as plt

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

def numTimeIntervals(numSamples, avgTime):
    # This function counts the number of exponentially-distributed time
    # intervals that occur in 1 time unit
    # numSamples: # of unit time intervals
    
    expSample = np.empty([numSamples])
    actualSample = 1
    
    while(actualSample <= numSamples - 1):
        countTimeIntervals = 0
        time = 0
        while(time <= 1):
            temp = exponentialRNG(1, avgTime)
            time = time + temp
            countTimeIntervals = countTimeIntervals+ 1
        actualSample = actualSample + 1
        expSample[actualSample-1] = countTimeIntervals
    
    return expSample
            

def main():
    avgTime = 0.2          # avgTime: average waiting time(lambda parameter)
    numSamples = 1000      # numSamples: # of samples to be generated

    expSample = numTimeIntervals(numSamples, avgTime)
    
    plt.figure(1)    
    plt.hist(expSample, bins = 'auto', color = 'blue', hatch = '/')
    plt.title("Histogram of counts of 1000 separate unit time intervals ", alpha = 0.75) 
    plt.xlabel('# of time intervals in 1 unit time interval')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig("Observed_Histogram_Intervals.png")
    plt.show()
    
if __name__ == "__main__":
    main()

