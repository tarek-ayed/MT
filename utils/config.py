from datasets.outlier_set import OutlierCIFAR, OutlierEasySet, OutlierMiniImageNet
from utils.outlier_detection import (
    call_with_pca,
    dbscan,
    elliptic,
    isolation_forest,
    knn,
    lof,
    svm,
)


DATASETS = {
    "CUB": OutlierEasySet(
        specs_file="data/CUB/test.json", training=False, image_size=224
    ),
    "CIFAR": OutlierCIFAR(
        "data/CIFAR/test.json",
        training=False,
        image_size=32,
        download=True,
        root="data/CIFAR",
    ),
    "MiniImageNet": OutlierMiniImageNet(
        root="/home/hawat/tarek/MT/data/miniImageNet/mini_imagenet",
        specs_file="/home/hawat/tarek/MT/data/miniImageNet/specs/test_images.csv",
        image_size=84,
    ),
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
