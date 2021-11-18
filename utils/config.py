from outlier_set import OutlierSet
from utils.outlier_detection import call_with_pca, dbscan, elliptic, isolation_forest, knn, lof, svm


DATASETS = {
    "CUB": OutlierSet(specs_file="./data/CUB/test.json", training=False),
    #"CIFAR": "./data/CIFAR/test.json",
}

OUTLIER_DETECTION_METHODS = {
    "DBSCAN": dbscan,
    "IsolationForest": isolation_forest,
    "SVM": svm,
    "LocalOutlierFactor": lof,
    "EllipticEnvelope": elliptic,
    "KNN": knn,
    "EllipticEnvelopePCA": call_with_pca(elliptic),
    "LocalOutlierFactorPCA": call_with_pca(lof),
    "IsolationForestPCA": call_with_pca(isolation_forest),
    "SVMPCA": call_with_pca(svm),
    "KNNPCA": call_with_pca(knn),
}
