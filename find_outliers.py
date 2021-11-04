#%%
from easyfsl.data_tools import task_sampler
from easyfsl.data_tools import EasySet
from torch.utils.data.sampler import BatchSampler
from outlier_set import OutlierSet
from torch.utils.data import DataLoader
import torch
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



model = torch.load("models/resnet50_pt_CUB_1st")

test_set = OutlierSet(specs_file="./data/CUB/test.json", training=False)
test_set.set_swaps(n_outliers=50)

N_CLASSES = 30

test_set.activate_outlier_mode()
class_loaders = [
    DataLoader(
        test_set,
        sampler=test_set.get_class(k)[0],  # images
        batch_size=64,
    )
    for k in range(N_CLASSES)
]
outlier_labels = [test_set.get_class(k)[1] for k in range(N_CLASSES)]

# %%
model.eval()
predictions = []
for class_ in range(N_CLASSES):

    counter = 0
    for imgs, _ in class_loaders[class_]:
        counter += 1
        imgs = imgs.cuda()
        z_backbone = model.backbone(imgs)
        scores = torch.cdist(z_backbone, z_backbone).detach().cpu().numpy()
    assert counter == 1

    clustering = DBSCAN(eps=0.6, min_samples=1, metric="precomputed").fit(
        scores / scores.max()
    )

    predictions.append(clustering.labels_ >= 1)

all_preds = np.concatenate(predictions, axis=0)
y_true = np.concatenate(outlier_labels, axis=0)

print(f"Accuracy: {accuracy_score(all_preds, y_true)}")
print(f"F1 Score: {f1_score(all_preds, y_true)}")
print(f"Precision: {precision_score(all_preds, y_true)}")
print(f"Recall: {recall_score(all_preds, y_true)}")