# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:38:24 2018

EE511 Project 3, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 2a.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import EM_algorithm as em
from scipy.stats import multivariate_normal
from matplotlib.colors import LogNorm
import timeit

def main():
    
    # generate spherical data centered on (10, 10)
    shifted_gaussian = []
    mu1 = np.array([10, 10])
    sigma1 = np.array([[3, 0], [0, 3]]) 
    for i in range(0,300):
        shifted_gaussian.append(multivariate_normal.rvs(mu1, sigma1))
        
    
    # generate [-5, -5] centered stretched Gaussian data (ellipsoidal)
    stretched_gaussian = []
    mu2 = np.array([-5, -5])
    sigma2 = np.array([[3, 0], [0, 1]]) 
    for i in range(0,300):
        stretched_gaussian.append(multivariate_normal.rvs(mu2, sigma2))
        
    # generated poorly distributed data
    poorly_distributed_gaussian = []
    mu3 = np.array([20, 20])
    sigma3 = np.array([[6, 3], [2, 4]]) 
    for i in range(0,300):
        poorly_distributed_gaussian.append(multivariate_normal.rvs(mu3, sigma3))
    
    # concatenate the two datasets into the final training set
    X_train = np.vstack([stretched_gaussian, shifted_gaussian])
    
    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(X_train)
    
    print("Parameters of the both the Gaussian distributions: ")
    print("\nMean 1: ",mu1)
    print("Covariance 1: ",sigma1)
    print("Mean 2: ",mu2)
    print("Covariance 2: ",sigma2)
    
    
    # display predicted scores by the model as a contour plot
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)
    
    plt.figure(4)
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)
    
    plt.title('Mixture of two Bivariate Gaussian distributions ')
    plt.axis('tight')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    data, label = clf.sample((300))
    
    data = np.asarray(data)
    numSamples = np.size(data, axis = 0)
    
    start = timeit.default_timer()
    wt1 = np.random.rand(np.size(data, axis = 0), 1)
    wt2 = np.empty([numSamples, 1])
    for i in range(0, np.size(data, axis = 0)):
        wt2[i] = 1 - wt1[i]
    weights = (wt1, wt2)
    parameters = em.Mstep(data, weights)

    loglikelihood = np.empty([500])
    for i in range(0, 500):
        #print("Iteration: ", i)
        (weights, loglike) = em.Estep(data, parameters)
        loglikelihood[i] = loglike
        parameters = em.Mstep(data, weights)
    
    stop = timeit.default_timer()
    print ("Total time taken for 500 iterations of EM (in seconds): ", stop - start)           
    param1, param2, param3, param4, param5 = parameters
    print("Final parameters of the mixture(calculated by the EM algorihtm): ")
    print("\nMean 1: ",param1)
    print("Covariance 1: ",param3)
    print("Mean 2: ",param2)
    print("Covariance 2: ",param4)
    print("Mixture weight: ", param5)
    
    
    label1 = np.empty([300])
    weights = np.asarray(weights)
    for i in range(0, 300):
        if (weights[0, i] > 0.5):
            label1[i] = 0
        else:
            label1[i] = 1
    #print(label1)
    
    plt.figure(3)
    plt.scatter(data[:, 0], data[:, 1], c= label1.reshape(300))
    plt.title(" Clustered mixture of bivariate Gaussians using EM algorithm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
if __name__ == "__main__":
    main()  