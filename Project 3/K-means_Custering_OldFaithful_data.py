# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:24:50 2018

EE511 Project 3, Prof. Osonde Osoba, Spring 2018
@author: Hrishikesh Hippalgaonkar
Tested in Python 3.6.3 :: Anaconda custom (64-bit), Windows 10

Question 1.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_table('Old_Faithful_2.txt', delim_whitespace=True, names=('#', 'Eruption Time', 'Waiting Time'))
data = df.values
data = data[:,(1,2)]

kmCluster = KMeans(n_clusters = 2, init= 'k-means++', random_state=10).fit(data)
cluster_labels = kmCluster.labels_
silhouette_avg = silhouette_score(data, cluster_labels, metric = 'euclidean')
print("Average silhouette score: ", silhouette_avg)

plt.figure(1)
plt.scatter(data[:,0], data[:, 1], c = cluster_labels, cmap='viridis')
plt.title("Scatter-plot for the 'Old_Faithful' data")
plt.xlabel("Eruption Time")
plt.ylabel("Waiting Time")
plt.grid(True, alpha = 0.3)

