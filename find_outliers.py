from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.parser import parse_args_outliers
from utils.config import DATASETS, OUTLIER_DETECTION_METHODS


args = parse_args_outliers()
print("------ Parameters for test_sparsity ------")
for parameter, value in args.__dict__.items():
    print(f"{parameter}: {value}")
print("------------------------------------------")


use_cuda = args.use_cuda
path_to_model = args.model_path
dataset = args.dataset
n_classes = args.n_classes
outlier_detection_methods = args.outlier_detection_methods
n_outliers = args.n_outliers

if use_cuda is None:
    use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

model = torch.load(path_to_model, map_location=torch.device(device))
model.eval()

test_set = DATASETS[dataset]
test_set.set_swaps(n_outliers=50)

test_set.activate_outlier_mode()
class_loaders = [
    DataLoader(
        test_set,
        sampler=test_set.get_class(k)[0],  # images
        batch_size=64,
    )
    for k in range(n_classes)
]
outlier_labels = [test_set.get_class(k)[1] for k in range(n_classes)]

predictions = []

for outlier_detection_name in outlier_detection_methods:

    detect_outliers = OUTLIER_DETECTION_METHODS[outlier_detection_name]

    for class_ in range(n_classes):

        counter = 0
        for imgs, _ in class_loaders[class_]:
            counter += 1
            if use_cuda:
                imgs = imgs.cuda()
            z_backbone = model.backbone(imgs)
            scores = torch.cdist(z_backbone, z_backbone).detach().cpu().numpy()
        assert counter == 1

        predictions.append(detect_outliers(scores))

    all_preds = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(outlier_labels, axis=0)

    print(
        f"Accuracy with {outlier_detection_name}: {accuracy_score(all_preds, y_true)}"
    )
    print(f"F1 Score with {outlier_detection_name}: {f1_score(all_preds, y_true)}")
    print(
        f"Precision with {outlier_detection_name}: {precision_score(all_preds, y_true)}"
    )
    print(f"Recall with {outlier_detection_name}: {recall_score(all_preds, y_true)}")
