# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:42:04 2018

EE511 Project #1, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2.

"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt

def BernoulliTrials(k):
    bernoulli_arr = np.empty([k])
    
    count = 0
    for i in range(0,k):
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
    
    success = np.empty([300])
    for i in range(0,300):
        success[i] = BernoulliTrials(50)
        
    plt.hist(success, bins = 'auto')
    plt.title("Histogram of 300 samples of success - counting random variable", alpha = 0.75)
    
    plt.xlabel('# of successes in 50 fair Bernoulli trials')
    plt.ylabel('# of samples')
    plt.grid(True)
    #plt.savefig("Success_in_5_Bernoulli.png")
    plt.show()
    
if __name__ == "__main__":
    main()