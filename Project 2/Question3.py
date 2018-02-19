# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:39:10 2018

EE511 Project #2, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 3.

"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt

def acceptReject(numSamples):
    
    
    expSample = np.empty([numSamples])
    rejectCount = 0 
    actualSamples = 0
    acceptSamples = 1

    while(acceptSamples <= numSamples):
        c = 1.4625
        flag = 1
        while(flag == 1):
            actualSamples = actualSamples + 1
            u1 = -6 * rand.random() + 6
            u2 = rand.random()
        
            if(u1 <= 1):
                f = 396 * pow((1 - u1), 4) * pow(u1, 7)
            elif(u1 > 4 and u1 <= 5):
                f = 0.5 * (u1 - 4)
            elif(u1 > 5 and u1 <=6):
                f = -0.5 * (u1 - 6)
            else:
                f = 0
            if(u2 <= (f/c)):
                expSample[acceptSamples - 1] = u1
                flag = 0
                acceptSamples = acceptSamples + 1
            else:
                flag = 1
                rejectCount = rejectCount + 1
            
            
    rejectionRate = (rejectCount/actualSamples) * 100
    print("\nTotal # of Samples: ", actualSamples)
    print("\n# of Rejected Samples: ", rejectCount)
    print("# of Accepted Samples: ", numSamples)
    print("\nRejection Rate: ", rejectionRate, " %")

    return (expSample)

def main():
    numSamples = 1000
    expSample = acceptReject(numSamples)
    plt.figure(1)
    plt.hist(expSample, bins = 'auto', hatch = '/')
    plt.title("Histogram of samples of the bimodal random variable", alpha = 0.75) 
    plt.xlabel('x ->')
    plt.ylabel('# of Samples')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
    

