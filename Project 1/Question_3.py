# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:58:50 2018

EE511 Project #1, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 3.

"""
import numpy as np
import random as rand
import matplotlib.pyplot as plt

def bernoulliTrialsForEdges(numPeople):

    edgeExist = []  # Empty list    
    count = 0

    for i in range(0, numPeople - 1):
        for j in range(i + 1, numPeople):
        # Return random floats in the half-open interval [0.0, 1.0).
        # Results are from the “continuous uniform” distribution over the stated interval
            x = rand.random()
            if(x <= 0.05):
                result = 1
                edgeExist.append([i, j, result])
                count = count + 1
            else:
                result = 0
                edgeExist.append([i, j, result])
    
    return count

def main():
    
    edgeExistTrial = np.empty([500])

    for i in range(0,500):
    	edgeExistTrial[i] = bernoulliTrialsForEdges(20)
            
    plt.hist(edgeExistTrial, bins = 'auto')
    plt.title("Histogram of 500 simulated trials for counting # of edges that exist in the social network", alpha = 0.75)
    
    plt.xlabel('# of edges in the social network')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()

