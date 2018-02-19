# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:35:10 2018

EE511 Project #1, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1.
 
"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import networkx as nx

def bernoulliTrialsForEdges(n, p):

    edgeExist = []  # Empty list    
    G = nx.Graph()
    for i in range(1, n):
        for j in range(i + 1, n + 1):
        # Return random floats in the half-open interval [0.0, 1.0).
        # Results are from the “continuous uniform” distribution over the stated interval
            x = rand.random()
            if(x <= p):
                result = 1
                edgeExist.append([i, j, result])
                G.add_edge(i, j)
            else:
                result = 0
                edgeExist.append([i, j, result])
                G.add_node(i)
                G.add_node(j)
    
    return G

def main():
    
    G = nx.Graph()
    n = 50  # Number of people in the network
    p = 0.12 # Probability of edge existing between two nodes
    degree = np.empty([n + 1])
    G = bernoulliTrialsForEdges(n, p)

    for i in range(1, n + 1):
        degree[i] = G.degree[i]

    print("Total # of edges",G.number_of_edges())
    plt.figure(1)
    plt.hist(degree, bins ='auto', alpha = 0.9, hatch ='/')
    plt.title("Histogram of degree of each node(p = 0.02)", alpha = 0.75)
    
    plt.xlabel('# of edges each node has in the network')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    plt.figure(2)
    plt.title("Graph for network of n = 50 , p = 0.02", alpha = 0.75)
    options = {
            'with_labels': True,
            'node_color': 'red',
            'node_size': 150,
            'width': 3
            }
    
    #nx.draw_circular(G, **options)
    #nx.draw_spectral(G, **options)
    nx.draw_random(G, **options)
    plt.show()
    
if __name__ == "__main__":
    main()
