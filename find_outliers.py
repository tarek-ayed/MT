#%%
from easyfsl.data_tools import task_sampler
from easyfsl.data_tools import EasySet
from outlier_set import OutlierSet
from torch.utils.data import DataLoader
#%%
import torch

model = torch.load('models/resnet18_pt_CUB_1st')

test_set = OutlierSet(specs_file="./data/CUB/test.json", training=False)
test_sampler = task_sampler(
    test_set, n_way=5, n_shot=5, n_query=10, n_tasks=100
)
test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

accuracy = model.evaluate(test_loader)
print(f"Average accuracy : {(100 * accuracy):.2f}")

#%%
test_set = OutlierSet(specs_file="./data/CUB/test.json", training=False)
test_set.set_swaps(n_outliers=50)
# %%
