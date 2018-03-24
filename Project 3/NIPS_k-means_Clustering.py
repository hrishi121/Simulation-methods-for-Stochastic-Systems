# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:00:28 2018

EE511 Project 3, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 3.

"""

from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import random

dataset = np.genfromtxt("nips-87-92.csv", delimiter = ",")
data = dataset[1:, 2:]
data_label = dataset[1:, 1]

svd = TruncatedSVD(n_components = 100)
 #For latent semantic analysis, value of 100 is recommended
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
data = lsa.fit_transform(data)

dist = []
x = []
silhouette_avg = []
#for i in range(2, 50):
    
#    kmCluster = KMeans(n_clusters= i, init='k-means++', max_iter=500).fit(data)
#    #kmCluster = MiniBatchKMeans(n_clusters = i, init='k-means++', n_init=1,
#    #                      batch_size=100, compute_labels = True).fit(data)
#    cluster_labels = kmCluster.labels_
#    dist.append(kmCluster.inertia_)
#    x.append(i)
#    temp = silhouette_score(data, cluster_labels, metric = 'euclidean')
#    silhouette_avg.append(temp)
#    print("For n_clusters =", i,
#      "The average silhouette_score is :", temp)
#
#silhouette_avg = np.asarray(silhouette_avg)
#index = np.argmax(silhouette_avg)
#print("The best silhouette score is: ",silhouette_avg[index], " for n_cluster :", index + 2)
#i = range(2, 50)
#plt.figure(1)
#plt.plot(i, silhouette_avg)
#plt.title("Silhouette score for different number of clusters")
#plt.ylabel("Silhouette score")
#plt.xlabel("Number of clusters")
#
#plt.figure(2)
#plt.plot(x, dist)
#plt.title("Plot of sum of squared distance of data samples from centroids")
#plt.ylabel("Sum of squared distance")
#plt.xlabel("Number of clusters")
    
kmCluster = KMeans(n_clusters= 2, init='k-means++', max_iter=1500).fit(data)
cluster_labels = kmCluster.labels_
for x in range(20):
  y = random.randint(0, 699)
  if(cluster_labels[y] == 1):
      print(data_label[y])