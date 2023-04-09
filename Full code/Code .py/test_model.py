import pytest
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score

# load data 
data_predict = joblib.load("D:/BOOTCAMP/project/(Block 4) ML Process/data feature/feature_normal.csv")

clust_normal = 9
# model Kmeans     
def kmeans(data, cluster):
    """
    Function for model kmeans clustering
    input data and sum of cluster
    """
    kmeans = KMeans(n_clusters=cluster) # input cluster
    kmeans.fit(data)
    
    ## Predictions
    y_pred = kmeans.predict(data)
    return  y_pred

data_predict['Cluster'] = kmeans(data = data_predict,cluster = clust_normal )

# cek best score centroid
def test_score():
    score = (round(silhouette_score(data_predict,data_predict['Cluster']),2))
    assert score == 1.0
