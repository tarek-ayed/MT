from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import torch


def dbscan(features_backbone):
    scores = torch.cdist(features_backbone, features_backbone).detach().cpu().numpy()
    clustering = DBSCAN(eps=0.6, min_samples=1, metric="precomputed").fit(
        scores / scores.max()
    )
    return clustering.labels_ >= 1, None


def isolation_forest(features_backbone):
    estimator = IsolationForest(n_estimators=500)
    return estimator.fit_predict(features_backbone) < 0, estimator.score_samples(features_backbone)


def svm(features_backbone):
    estimator = OneClassSVM().fit(features_backbone)
    return estimator.fit_predict(features_backbone) < 0, estimator.score_samples(features_backbone)


def lof(features_backbone):
    estimator = LocalOutlierFactor().fit(features_backbone)
    return estimator.fit_predict(features_backbone) < 0, estimator.negative_outlier_factor_
