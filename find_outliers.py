from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tqdm

from utils.parser import parse_args_outliers
from utils.config import DATASETS, OUTLIER_DETECTION_METHODS


args = parse_args_outliers()
print("------ Parameters for find_outliers ------")
for parameter, value in args.__dict__.items():
    print(f"{parameter}: {value}")
print("------------------------------------------")


use_cuda = args.use_cuda
path_to_model = args.model_path
dataset = args.dataset
outlier_detection_methods = args.outlier_detection_methods
proportion_outliers = args.proportion_outliers
num_samples = args.n_samples
n_shot = args.n_shot

if use_cuda is None:
    use_cuda = torch.cuda.is_available()
if use_cuda and args.gpu_to_use:
    torch.cuda.set_device(args.gpu_to_use)
device = "cuda" if use_cuda else "cpu"

torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
model = torch.load(path_to_model, map_location=torch.device(device))
model.eval()

test_set = DATASETS[dataset]
print("Computing backbone features ...")
test_set.set_model(model, use_cuda)
print("Done.")

outlier_labels = []
predictions = dict(
    (outlier_detection_name, []) for outlier_detection_name in outlier_detection_methods
)
predictions_scores = dict(
    (outlier_detection_name, []) for outlier_detection_name in outlier_detection_methods
)

print("Computing outlier detection predictions...")
for _ in tqdm.tqdm(range(num_samples)):

    features_backbone, labels = test_set.sample_class_features_with_outliers(
        proportion_outliers=proportion_outliers, limit_num_samples=n_shot
    )
    outlier_labels.append(labels)

    for outlier_detection_name in outlier_detection_methods:

        detect_outliers = OUTLIER_DETECTION_METHODS[outlier_detection_name]
        predicted_labels, predicted_scores = detect_outliers(features_backbone)

        predictions[outlier_detection_name].append(predicted_labels)

        if predicted_scores is not None:
            predictions_scores[outlier_detection_name].append(predicted_scores)

y_true = np.concatenate(outlier_labels, axis=0)

for outlier_detection_name in outlier_detection_methods:

    all_preds = np.concatenate(predictions[outlier_detection_name], axis=0)

    print(
        f"Accuracy with {outlier_detection_name}: {accuracy_score(all_preds, y_true):.4f}"
    )
    print(f"F1 Score with {outlier_detection_name}: {f1_score(all_preds, y_true):.4f}")
    print(
        f"Precision with {outlier_detection_name}: {precision_score(all_preds, y_true):.4f}"
    )
    print(
        f"Recall with {outlier_detection_name}: {recall_score(all_preds, y_true):.4f}"
    )
    if len(predictions_scores[outlier_detection_name]) > 0:
        all_scores = np.concatenate(predictions_scores[outlier_detection_name], axis=0)
        try:
            auc_score = roc_auc_score(y_true, all_scores)
            print(f"ROC AUC with {outlier_detection_name}: {auc_score:.4f}")
            fpr, tpr, _ = roc_curve(y_true, all_scores)
            plt.clf()
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
            plt.title(f"ROC Curve using {outlier_detection_name} on {dataset}")
            plt.savefig(f"ROC_{outlier_detection_name}")
        except ValueError:
            print(f"Unable to compute ROC Curve for {outlier_detection_name}")
    print("------------------------------------------")
