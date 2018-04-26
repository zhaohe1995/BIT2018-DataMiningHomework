# -*- coding:UTF-8 -*-
"""
3 clusters
@author: ZhaoHe
"""
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

def cluster_KMeans(train_X, test_X):
    clf = KMeans(n_clusters=2)
    clf.fit(train_X)
    predictions = clf.predict(test_X)
    return predictions

def cluster_Hierarchical(test_X):
    clf = AgglomerativeClustering(n_clusters=2)
    predictions = clf.fit_predict(test_X)
    return predictions

def cluster_Spectral(test_X):
    clf = SpectralClustering(n_clusters=2)
    predictions = clf.fit_predict(test_X)
    return predictions