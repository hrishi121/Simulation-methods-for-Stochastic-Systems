# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:51:44 2018

EE511 Project 3, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2.

"""

import pandas as pd
import numpy as np
import EM_algorithm as em
import matplotlib.pyplot as plt
    
def main():

    df = pd.read_table('Old_Faithful_2.txt', delim_whitespace=True, names=('#', 'Eruption Time', 'Waiting Time'))
    data = df.values
    data = data[:,(1,2)]
    numSamples = np.size(data, axis = 0)
    
    wt1 = np.random.rand(np.size(data, axis = 0), 1)
    wt2 = np.empty([numSamples, 1])
    for i in range(0, np.size(data, axis = 0)):
        wt2[i] = 1 - wt1[i]
    weights = (wt1, wt2)
    parameters = em.Mstep(data, weights)

    loglikelihood = np.empty([1000])
    for i in range(0, 1000):
        #print("Iteration: ", i)
        (weights, loglike) = em.Estep(data, parameters)
        loglikelihood[i] = loglike
        parameters = em.Mstep(data, weights)
                
    param1, param2, param3, param4, param5 = parameters
    print("Final parameters: ")
    print("\nMean 1: ",param1)
    print("Covariance 1: ",param3)
    print("Mean 2: ",param2)
    print("Covariance 2: ",param4)
    print("Mixture weight: ", param5)
    
    em.plotDecisionRegion(data, parameters)
    
#    plt.figure(2)
#    x = np.linspace(0, 20, 20)
#    plt.plot(x, loglikelihood)
#    plt.title('Log Likelihood')
    
    label1 = np.empty([numSamples])
    weights = np.asarray(weights)
    for i in range(0, numSamples):
        if (weights[0, i] > 0.5):
            label1[i] = 0
        else:
            label1[i] = 1
    #print(label1)
    plt.figure(3)
    plt.scatter(data[:, 0], data[:, 1], c= label1.reshape(numSamples))
    
if __name__ == "__main__":
    main()                                                                      