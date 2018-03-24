# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 20:16:31 2018

EE511 Project 3, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import log
from matplotlib.colors import LogNorm


"Calculate probability of a data point given the current parameters"
def MultivariateGaussianPDF(datum, mu1, sigma1):
    
    sigma = np.asarray(sigma1)
    mu = np.asarray(mu1).reshape(2)
    data = np.asarray(datum).reshape(2)
    y = multivariate_normal.pdf(data, mean = mu, cov = sigma)
    return y

"The Estimation Step"
def Estep(data, parameters):
    mu1, mu2, sigma1, sigma2, mixWeight = parameters
    i = 0
    loglike = 0
    wt1 = np.empty([np.size(data, axis = 0), 1])
    wt2 = np.empty([np.size(data, axis = 0), 1])
    for dataSample in data:
        # unnormalized weights (or "responsibility")
        wt1[i] = MultivariateGaussianPDF(dataSample, mu1, sigma1) * mixWeight
        wt2[i] = MultivariateGaussianPDF(dataSample, mu2, sigma2) * (1. - mixWeight)
        # compute denominator
        den = wt1[i] + wt2[i]
        # normalize
        wt1[i] = wt1[i] / den
        wt2[i] = wt2[i] / den
        loglike = loglike + log(wt1[i] + wt2[i])
        i = i + 1

    # Return the weight tuple
    weights = (wt1, wt2)
    #print("E Step done!")
    return (weights, loglike)

"The Maximation Step"
def Mstep(data1, weights):
    wt11, wt22 = weights
    wt1 = np.asarray(wt11)
    wt2 = np.asarray(wt22)
    data = np.asarray(data1)
    totalDataSamples = np.size(data, axis = 0)
    totalWeight1 = sum(wt1)      #This is the total responsibility allocated to cluster 1
    totalWeight2 = sum(wt2)      #This is the total responsibility allocated to cluster 2
    mixWeight = totalWeight1 / totalDataSamples
    
    # Update the means mu1 and mu2
    feature1 = np.reshape(data[:, 0], (totalDataSamples, 1))
    feature2 = np.reshape(data[:, 1], (totalDataSamples, 1))
    
    t1 = (sum(wt1 * feature1)) / totalWeight1
    t2 = (sum(wt1 * feature2)) / totalWeight1
    t3 = (sum(wt2 * feature1)) / totalWeight2
    t4 = (sum(wt2 * feature2)) / totalWeight2
    
    mu1 = np.array([t1, t2])
    mu2 = np.array([t3, t4])

    
    # Update the covariance matrices sigma1 an sigma2
    sigma1 = np.zeros([2, 2])
    sigma2 = np.zeros([2, 2])
    i = 0
    for datum in data:
        dataSample = np.asarray(datum).reshape(2,1)
        y1 = dataSample - mu1
        y2 = dataSample - mu2
        sigma1 = sigma1 + wt1[i] * np.outer(np.transpose(y1), y1)
        sigma2 = sigma2 + wt2[i] * np.outer(np.transpose(y2), y2)
        i = i + 1

    sigma1 = sigma1 / totalWeight1
    sigma2 = sigma2 / totalWeight2  

    parameters = (mu1, mu2, sigma1, sigma2, mixWeight)
    #print("M Step done!")
    return parameters
    
"Plot the Decision Region using the parameters from EM computation"
def plotDecisionRegion(data, parameters):
    
    param1, param2, param3, param4, param5 = parameters
    
    # Make the plot
    u1 = np.linspace(param1[0] - 3, param1[0] + 3, 1000)
    v1 = np.linspace(param1[1] - 30, param1[1] + 30, 1000)
    u2 = np.linspace(param2[0] - 3, param2[0] + 3, 1000)
    v2 = np.linspace(param2[1] - 30, param2[1] + 30, 1000)

    x1, y1 = np.meshgrid(u1, v1)
    data1 = np.dstack((x1, y1))

    x2, y2 = np.meshgrid(u2, v2)
    data2 = np.dstack((x2, y2))

    sigma1 = np.asarray(param3)
    mu1 = np.asarray(param1).reshape(2)
    sigma2 = np.asarray(param4)
    mu2 = np.asarray(param2).reshape(2)
     
    z1 = multivariate_normal.pdf(data1, mean = mu1, cov = sigma1)
    z2 = multivariate_normal.pdf(data2, mean = mu2, cov = sigma2)
    
    plt.figure(1)
    plt.plot(data[:, 0], data[:, 1], "r+")
    plt.contour( x1, y1, z1 )
    plt.contour( x2, y2, z2 )
    plt.title("Clustering using Gaussian Mixture Model and EM algorithm", alpha = 0.75)
    plt.legend('Data Samples', loc = 4)