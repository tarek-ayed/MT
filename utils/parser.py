import argparse


def parse_args_outliers():

    parser = argparse.ArgumentParser(description="Argument parser for find_outliers.py")

    parser.add_argument("--dataset", default="CUB", type=str, help="Dataset to use")

    parser.add_argument(
        "--model_path",
        default="models/resnet50_pt_CUB_1st",
        type=str,
        help="Path to the model to use",
    )

    parser.add_argument(
        "--n_classes", default=30, type=int, help="Number of classes in the dataset"
    )

    parser.add_argument(
        "--use_cuda",
        default=None,
        type=bool,
        help="Specify whether to use GPU. By default uses it when available",
    )

    parser.add_argument(
        "--outlier_detection_methods",
        default=["DBSCAN", "IsolationForest", "LocalOutlierFactor", "KNN"],
        type=str,
        nargs="*",
        help="Specify which outlier detection methods to use",
    )

    parser.add_argument(
        "--n_outliers",
        default=50,
        type=int,
        help="Number of outliers to add in dataset",
    )

    return parser.parse_args()
