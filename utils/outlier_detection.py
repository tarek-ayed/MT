from sklearn import cluster
from sklearn.cluster import DBSCAN

def dbscan(scores):
    clustering = DBSCAN(eps=0.6, min_samples=1, metric="precomputed").fit(
        scores / scores.max()
    )
    return clustering.labels_ >= 1