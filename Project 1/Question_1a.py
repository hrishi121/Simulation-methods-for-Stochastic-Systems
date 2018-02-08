# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:58:06 2018

EE511 Project #1, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1a.

"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt

def main():
        
    bernoulli_arr = np.empty([100])
    
    for i in range(0,100):
    # Return random floats in the half-open interval [0.0, 1.0).
    # Results are from the “continuous uniform” distribution over the stated interval
        x = rand.random()
        if(x >= 0.5):
            bernoulli_arr[i] = 1
        else:
            bernoulli_arr[i] = 0
            
    plt.hist(bernoulli_arr, bins = 3)
    plt.title("Histogram of 100 fair Bernoulli trials", alpha = 0.75)
    
    plt.xlabel('Output of the Bernoulli Trial')
    plt.ylabel('# of trials')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
