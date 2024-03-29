from pathlib import Path

import torch
from PIL import Image
from typing import Callable, Optional

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MiniImageNet(VisionDataset):
    def __init__(
        self,
        root: str,
        specs_file: str,
        image_size: int = 84,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):

        self.loading_transform = transforms.Compose(
            [
                transforms.Resize([int(image_size * 1.4), int(image_size * 1.4)]),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )

        transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                ]
            )
        )

        super(MiniImageNet, self).__init__(
            str(root), transform=transform, target_transform=target_transform
        )

        # Get images and labels
        data_df = pd.read_csv(specs_file).assign(
            image_paths=lambda df: df.apply(
                lambda row: Path(root) / row["class_name"] / row["image_name"], axis=1
            )
        )
        self.images = torch.stack(
            [
                self.load_image_as_tensor(image_path)
                for image_path in tqdm(data_df.image_paths)
            ]
        )

        self.class_list = data_df.class_name.unique()
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.labels = list(data_df.class_name.map(self.class_to_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        img, label = (
            self.transform(self.images[item]),
            self.labels[item],
        )

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def load_image_as_tensor(self, filename):
        return self.loading_transform(Image.open(filename).convert("RGB"))