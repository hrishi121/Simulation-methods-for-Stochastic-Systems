# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:25:12 2018

EE511 Project #1, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1b.

"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt

def sevenBernoulliTrials():
    bernoulli_arr = np.empty([7])
    
    count = 0
    for i in range(0,7):
    # Return random floats in the half-open interval [0.0, 1.0).
    # Results are from the “continuous uniform” distribution over the stated interval
        x = rand.random()
        if(x >= 0.5):
            bernoulli_arr[i] = 1
            count = count + 1
        else:
            bernoulli_arr[i] = 0
    
    return count

def main():
    
    success = np.empty([100])
    for i in range(0,100):
        success[i] = sevenBernoulliTrials()
        
    plt.hist(success, bins = 'auto')
    plt.title("Histogram of 100 samples of success - counting random variable", alpha = 0.75)
    
    plt.xlabel('# of successes in 7 fair Bernoulli trials')
    plt.ylabel('# of samples')
    plt.grid(True)
    plt.savefig("Success_in_7_Bernoulli.jpg")
    plt.show()
    
if __name__ == "__main__":
    main()
            

