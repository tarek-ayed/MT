#%%
import easyfsl as fsl
import os

from easyfsl.data_tools import EasySet, TaskSampler
from torch.utils.data import DataLoader

from easyfsl.methods import PrototypicalNetworks
import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet18

GPU_TO_USE = 0
torch.cuda.set_device(GPU_TO_USE)

#%%
train_set = EasySet(specs_file="./data/CUB/train.json", training=True)
train_sampler = TaskSampler(
    train_set, n_way=5, n_shot=5, n_query=10, n_tasks=40000
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

#%%
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
model = PrototypicalNetworks(convolutional_network).cuda()

optimizer = Adam(params=model.parameters())

model.fit(train_loader, optimizer)

#%%
test_set = EasySet(specs_file="./data/CUB/test.json", training=False)
test_sampler = TaskSampler(
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