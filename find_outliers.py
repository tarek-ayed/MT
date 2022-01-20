import torch
import numpy as np
import tqdm

from utils.parser import parse_args_outliers
from utils.config import DATASETS, OUTLIER_DETECTION_METHODS
from utils.metrics import print_metrics


args = parse_args_outliers()
print("------ Parameters for find_outliers ------")
for parameter, value in args.__dict__.items():
    print(f"{parameter}: {value}")
print("------------------------------------------")


device = args.device
path_to_model = args.model_path
dataset = args.dataset
outlier_detection_methods = args.outlier_detection_methods
proportion_outliers = args.proportion_outliers
num_samples = args.n_samples
n_shot = args.n_shot
num_classes = args.num_classes

if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
model = torch.load(path_to_model, map_location=torch.device(device))
model.eval()

test_set = DATASETS[dataset]()
print("Computing backbone features ...")
test_set.set_model(model, device)
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
        proportion_outliers=proportion_outliers,
        limit_num_samples=n_shot,
        num_classes=num_classes,
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

    print_metrics(
        dataset, predictions_scores, outlier_detection_name, y_true, all_preds
    )
