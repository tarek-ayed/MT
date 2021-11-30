import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import torch
from pyod.models.knn import KNN


def to_numpy(vectors):
    if type(vectors) is np.ndarray:
        return vectors
    return vectors.detach().cpu().numpy()


def dbscan(features_backbone):
    features_backbone = torch.tensor(features_backbone)
    scores = to_numpy(torch.cdist(features_backbone, features_backbone))
    clustering = DBSCAN(eps=0.6, min_samples=1, metric="precomputed").fit(
        scores / scores.max()
    )
    return clustering.labels_ >= 1, None


def isolation_forest(features_backbone):
    features_backbone = to_numpy(features_backbone)
    estimator = IsolationForest(n_estimators=500).fit(features_backbone)
    return estimator.predict(features_backbone) < 0, -estimator.score_samples(
        features_backbone
    )


def svm(features_backbone):
    features_backbone = to_numpy(features_backbone)
    estimator = OneClassSVM().fit(features_backbone)
    return estimator.predict(features_backbone) < 0, -estimator.score_samples(
        features_backbone
    )


def lof(features_backbone):
    features_backbone = to_numpy(features_backbone)
    estimator = LocalOutlierFactor()
    predictions = estimator.fit_predict(features_backbone)
    return predictions < 0, -estimator.negative_outlier_factor_


def elliptic(features_backbone):
    features_backbone = to_numpy(features_backbone)
    estimator = EllipticEnvelope().fit(features_backbone)
    return estimator.predict(features_backbone) < 0, -estimator.score_samples(
        features_backbone
    )


def knn(features_backbone):
    features_backbone = to_numpy(features_backbone)
    estimator = KNN(n_neighbors=min(4, len(features_backbone) - 1)).fit(
        features_backbone
    )
    return (
        estimator.predict(features_backbone),
        estimator.predict_proba(features_backbone)[:, 1],
    )


def call_with_pca(outlier_detection_func):
    def func_with_pca(features_backbone):
        features_backbone = PCA(n_components=None).fit_transform(
            to_numpy(features_backbone)
        )
        # n_components set to None means taking min(n_samples, n_features)
        return outlier_detection_func(features_backbone)

    return func_with_pca
