from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def dbscan(scores):
    clustering = DBSCAN(eps=0.6, min_samples=1, metric="precomputed").fit(
        scores / scores.max()
    )
    return clustering.labels_ >= 1, None


def isolation_forest(scores):
    estimator = IsolationForest(n_estimators=500)
    return estimator.fit_predict(scores) < 0, estimator.score_samples(scores)


def svm(scores):
    estimator = OneClassSVM().fit(scores)
    return estimator.fit_predict(scores) < 0, estimator.score_samples(scores)


def lof(scores):
    estimator = LocalOutlierFactor().fit(scores)
    return estimator.fit_predict(scores) < 0, estimator.negative_outlier_factor_
