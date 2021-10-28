#%%
from easyfsl.data_tools import task_sampler
from easyfsl.data_tools import EasySet
from torch.utils.data.sampler import BatchSampler
from outlier_set import OutlierSet
from torch.utils.data import DataLoader
import torch
from sklearn.cluster import DBSCAN


model = torch.load("models/resnet18_pt_CUB_1st")

test_set = OutlierSet(specs_file="./data/CUB/test.json", training=False)
test_set.set_swaps(n_outliers=50)
outlier_labels = test_set.get_outlier_labels()

N_CLASSES = 30

class_loaders = [
    DataLoader(
        test_set,
        sampler=test_set.get_class(k), 
        batch_size=64,
    )
    for k in range(N_CLASSES)
]
# %%
model.eval()
predictions = []
for class_ in range(N_CLASSES):
    counter=0
    for imgs, _ in class_loaders[class_]:
        counter += 1
        imgs = imgs.cuda()
        z_backbone = model.backbone(imgs)
        scores = torch.cdist(z_backbone, z_backbone).detach().cpu().numpy()
    assert counter==1
    clustering = DBSCAN(eps=0.6, min_samples=1, metric="precomputed").fit(scores/scores.max())
    predictions += list(clustering.labels_)
# %%
import matplotlib.pyplot as plt

plt.pcolormesh(scores.cpu().detach().numpy())
# %%
