from outlier_set import OutlierSet
from utils.outlier_detection import dbscan, isolation_forest, lof, svm


DATASETS = {
    "CUB": OutlierSet(specs_file="./data/CUB/test.json", training=False),
    #"CIFAR": "./data/CIFAR/test.json",
}

OUTLIER_DETECTION_METHODS = {
    "DBSCAN": dbscan,
    "IsolationForest": isolation_forest,
    "SVM": svm,
    "LocalOutlierFactor": lof,
}
