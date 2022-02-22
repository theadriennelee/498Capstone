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

skipVal = 11840
midVal = 15787

def set_validation_data(index):
    """
    Helper function to read validation data

    Args:
        index (int): Index of datapoint
    Returns:
        boolean: Returns false if datapoint is a validation point
    """
    if skipVal < index < midVal:
        return False
    elif index == 0:
        return False
    else:
        return True

def set_testing_data(index):
    """
    Helper function to read validation data

    Args:
        index (int): Index of datapoint
    Returns:
        boolean: Returns false if datapoint is a validation point
    """
    if index >= midVal:
        return False
    elif index == 0:
        return False
    else:
        return True


# LOOK AT ADRIENNE'S PREDICTED_INVALID.CSV TO RECREATE YOUR OWN WITH YOUR DATA
# SHOULD INCLUDE TIMESTAMPS (DATE), INDEX AND T/F FLAG
filename = "./CNN/predicted_invalid.csv"
dataset_filename = "zero_T1n.csv"
data = pd.read_csv(filename, usecols=['Unnamed: 0', '1'])
timestamps = pd.read_csv(filename, usecols=['0'])
timestamps = timestamps.values
X = data.to_numpy()

# CURRENTLY SET TO READ VALIDATION DATA, CHANGE TO set_testing_data IF RUNNING ON TESTING DATA
series = pd.read_csv(dataset_filename, sep=',', header=0, index_col=0, usecols=['Unnamed: 0', 'T1_class'], squeeze=True, skiprows=lambda x: set_testing_data(x))
# data_timestamps = pd.read_csv(dataset_filename, sep=',', header=0, index_col=0, usecols=['date'], squeeze=True, skiprows=lambda x: set_validation_data(x))
data_timestamps = pd.read_csv(dataset_filename, usecols=['date'], skiprows=lambda x: set_testing_data(x))
data_timestamps = data_timestamps.values
test_dataset = pd.DataFrame()
i = 0
j = 0
for flag in series.values:
    if flag == True:
        df = pd.DataFrame([[i, series.index[j] - skipVal - 1]])
        test_dataset = test_dataset.append(df)
        i = i + 1
    j = j + 1
Y = test_dataset.to_numpy()

        


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
     pyplot.scatter(X[row_ix, 1], X[row_ix, 0])
pyplot.show()

for cluster in clusters:
     # get row indexes for samples with this cluster
     row_ix = where(yhat == cluster)
     # create scatter of these samples
     pyplot.scatter(X[row_ix, 1], X[row_ix, 0], c = ["red"], label="Predicted")

# fit model and predict clusters
model_data = DBSCAN(eps=30, min_samples=15)
data_yhat = model_data.fit_predict(Y)
# retrieve unique clusters
data_clusters = unique(data_yhat)

data_start_of_anomaly = []
data_end_of_anomaly = []
data_timestamp_anomaly = []

# create scatter plot for samples from each cluster
for cluster in data_clusters:
     # get row indexes for samples with this cluster
     row_ix = where(data_yhat == cluster)
     data_start_of_anomaly.append(X[row_ix[0][0], 0])
     data_end_of_anomaly.append(X[row_ix[0][len(row_ix[0]) - 1], 0])
     # create scatter of these samples
     pyplot.scatter(Y[row_ix, 1], Y[row_ix, 0], c = ["blue"], label="Actual")


# show the plot
pyplot.show()

for i in range(1, len(start_of_anomaly), 1):
    timestamp_anomaly.append([timestamps[start_of_anomaly[i]], timestamps[end_of_anomaly[i]]])

for i in range(1, len(data_start_of_anomaly), 1):
    data_timestamp_anomaly.append([data_timestamps[data_start_of_anomaly[i]], data_timestamps[data_end_of_anomaly[i]]])

# LOOK FOR TIMESTAMP_ANOMALY TO SEE THE START AND END POINTS OF CLUSTER
# TWO PLOTS SHOULD BE CREATED
# THE FIRST PLOT IS THE CLUSTERS ON PREDICTED DATA - PLAY AROUND WITH EPS AND MIN_SAMPLE TO FIND CORRECT CLUSTER
# THE SECOND PLOT IS A COMPARISON BETWEEN PREDICTED DATA CLUSTERS AND ACTUAL CLUSTER




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


