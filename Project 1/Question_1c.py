# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:29:57 2018

EE511 Project #1, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1c.

"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt

def bernoulliTrials(numTrials):
    bernoulli_arr = np.empty([numTrials])
    
    for i in range(0,numTrials):
    # Return random floats in the half-open interval [0.0, 1.0).
    # Results are from the “continuous uniform” distribution over the stated interval
        x = rand.random()
        if(x >= 0.5):
            bernoulli_arr[i] = 1
        else:
            bernoulli_arr[i] = 0
    
    return bernoulli_arr

def countLongestRun(bernoulli_arr):

    longestRun = 1
    count = 1

    for i in range(0, np.size(bernoulli_arr) - 1):
        if(bernoulli_arr[i] == 1 and bernoulli_arr[i + 1] == 1):
            count = count + 1
        else:
            count = 1

        if(count >= longestRun):
            longestRun = count
        else:
            longestRun = longestRun

    return longestRun

def main():

    longestRun = np.empty([400])

    for i in range(0,400):
        bernoulli_arr = bernoulliTrials(100)
        longestRun[i] = countLongestRun(bernoulli_arr)

    plt.hist(longestRun, bins = 'auto')
    plt.title("Histogram of 400 trials of 'counting longest run of heads in 100 Bernoulli Trials ", alpha = 0.75)
    
    plt.xlabel('Longest Run of Heads')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()