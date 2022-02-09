# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:51:09 2022

@author: Adrienne Lee
"""

import pandas as pd
import matplotlib.pyplot as plt

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from matplotlib import pyplot


# var = pd.read_csv("./clusteringData/predicted_invalid.csv")
filename = "predicted_invalid.csv"
data = pd.read_csv(filename, usecols=['0', '2'])
timestamps = pd.read_csv(filename, usecols=['1'])
timestamps = timestamps.values
X = data.to_numpy()
# print(var)

# x = list(var['0'])
# y = list(var['2'])

# plt.figure(figsize=(10,10))
# plt.style.use('seaborn')
# plt.scatter(x,y,marker="*",s=100,edgecolors="black",c="yellow")
# plt.title("Excel sheet to Scatter Plot")
# plt.show()

# series = pd.read_csv("./clusteringData/predicted_invalid.csv", 
#                       sep=',', 
#                       header=0, 
#                       index_col=0, 
#                       squeeze=True)

# series.reset_index().to_numpy()

# AFFINITY
# define the model
# validation works for 0.7
# model = AffinityPropagation(damping=0.9)
# # fit the model
# model.fit(X)
# # assign a cluster to each example
# yhat = model.predict(X)
# # retrieve unique clusters
# clusters = unique(yhat)

# # hold starts and ends of anomalies
# start_of_anomaly = []
# end_of_anomaly = []

# # create scatter plot for samples from each cluster
# for cluster in clusters:
#     # get row indexes for samples with this cluster
#     row_ix = where(yhat == cluster)
#     # append start and end of each cluster
#     start_of_anomaly.append([X[row_ix[0][0], 0], X[row_ix[0][0], 1]])
#     end_of_anomaly.append([X[row_ix[0][len(row_ix[0]) - 1], 0], X[row_ix[0][len(row_ix[0]) - 1], 0]])
#     # create scatter of these samples
#     pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()

# DBSCAN
# not bad for both, clusters the outliers together so maybe if we can filter those out
# define dataset
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = DBSCAN(eps=30, min_samples=15)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)

start_of_anomaly = []
end_of_anomaly = []
timestamp_anomaly = []

# create scatter plot for samples from each cluster
for cluster in clusters:
     # get row indexes for samples with this cluster
     row_ix = where(yhat == cluster)
     start_of_anomaly.append(X[row_ix[0][0], 0])
     end_of_anomaly.append(X[row_ix[0][len(row_ix[0]) - 1], 0])
     # create scatter of these samples
     pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()

for i in range(1, len(start_of_anomaly), 1):
    timestamp_anomaly.append([timestamps[start_of_anomaly[i]], timestamps[end_of_anomaly[i]]])

# # MEAN SHIFT
# # not exactly the best for testing
# # define the model
# model = MeanShift()
# # fit model and predict clusters
# yhat = model.fit_predict(X)
# # retrieve unique clusters
# clusters = unique(yhat)
# # create scatter plot for samples from each cluster
# for cluster in clusters:
#      # get row indexes for samples with this cluster
#      row_ix = where(yhat == cluster)
#      # create scatter of these samples
#      pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()

# # OPTICS
# # not bad for both
# # define the model
# model = OPTICS(eps=10, min_samples=28)
# # fit model and predict clusters
# yhat = model.fit_predict(X)
# # retrieve unique clusters
# clusters = unique(yhat)
# # create scatter plot for samples from each cluster
# for cluster in clusters:
#      # get row indexes for samples with this cluster
#      row_ix = where(yhat == cluster)
#      # create scatter of these samples
#      pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()


