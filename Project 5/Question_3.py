# -*- coding: utf-8 -*-
"""
Created on Wed May  2 22:42:44 2018

EE511 Project #5, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10
Tested in Python 3.5.4 :: Anaconda custom (64-bit), Windows 10

Question 3.
Markov Chain Monte Carlo: Travelling Salesman Problem using Simulated Annealing
"""


import numpy as np
import math
import copy
import sys
import time
from math import exp, log
import matplotlib.pyplot as plt
import random


def updt(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rRuns: [{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def t1(i):
    # Cooling function : logarithmic
    t = 500000/(1 + log(i + 1))
    return t

def t2(i):
    # Cooling function : exponential
    t = 0.8*i
    return t

def t3(i):
    # Cooling function : linear
    t = 500000/(1 + 2*i)
    return t

def t4(i):
    # Cooling function : quadratic
    t = 500000/(1 + 2*i*i)
    return t


def pos_swap(city, loc1, loc2):
    temp_city = copy.copy(city)  # This creates a shallow copy
    temp_city[[loc1, loc2]] = temp_city[[loc2, loc1]]
    return temp_city

def distance(city):
    dist = 0
    loc = city[:, (0,1)]
    for i in range (1, len(city)):
        dist = dist + np.linalg.norm(loc[i-1,:] - loc[i,:])
    return dist


def main():
    
    #t_curr = 50
    
    city = np.genfromtxt('cities_400.csv', delimiter = ',')
    n = np.linspace(1,400,num = 400); n1 = np.reshape(n,(400,1))
    city = np.hstack((city,n1))    
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim([0,1000])
    ax.set_xlim([0,1000])
    plt.title('Travelling salesman problem')
    
    
    for i in range(0,8000):
        
        #updt(1000, i + 1)
        
#        if (i%50 == 0):
#            t_curr = t2(t_curr)
        
        t_curr = t3(i)
        
        pos1 = random.randint(0,399)
        pos2 = random.randint(0,399)
        
        dist_prev = distance(city)
        temp_city = pos_swap(city, pos1, pos2)
        dist_new = distance(temp_city)
        if i == 0:
            min_dist = dist_prev
            
        u = np.random.uniform()
        delta = dist_new - dist_prev
        ratio = np.exp(-delta/t_curr)
        alpha = min(1, ratio)
        
        if((delta < 0) or (u <= alpha)):
            city = temp_city
            min_dist = dist_new
        if(i%50 == 0):
            print('Epoch:{0}\t Temperature:{1:0.2f} Minimum distance:{2:0.2f}'.format(i, t_curr, min_dist))
        
        time.sleep(0.0001)
        plt.cla()
        ax.plot(city[:,0], city[:,1])
        for q, txt in enumerate(n):
            ax.annotate(txt, (city[q,0],city[q,1]))
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    print('\nFinal City Tour:\n')
    print(city)

    
if __name__ == "__main__":
    main()