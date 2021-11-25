from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
n_classes = args.n_classes
outlier_detection_methods = args.outlier_detection_methods
proportion_outliers = args.proportion_outliers

if use_cuda is None:
    use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
model = torch.load(path_to_model, map_location=torch.device(device))
model.eval()

test_set = DATASETS[dataset]
test_set.set_swaps(proportion_outliers=proportion_outliers)

test_set.activate_outlier_mode()

classes_to_fetch = set(test_set.labels)
class_loaders = {
    class_: DataLoader(
        test_set,
        sampler=test_set.get_class(class_),  # image indices
        batch_size=64,
    )
    for class_ in classes_to_fetch
}
outlier_labels = [
    test_set.get_class_outlier_labels(class_) for class_ in classes_to_fetch
]

predictions = dict(
    (outlier_detection_name, []) for outlier_detection_name in outlier_detection_methods
)
predictions_scores = dict(
    (outlier_detection_name, []) for outlier_detection_name in outlier_detection_methods
)

for class_ in classes_to_fetch:

    counter = 0
    features_backbone_list = []
    for imgs, _ in class_loaders[class_]:
        counter += 1
        if use_cuda:
            imgs = imgs.cuda()
        features_backbone_list.append(model.backbone(imgs))
    features_backbone = torch.cat(features_backbone_list)

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
        print(
            f"ROC AUC with {outlier_detection_name}: {roc_auc_score(y_true, all_scores):.4f}"
        )
    print("------------------------------------------")
