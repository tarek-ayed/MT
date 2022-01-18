import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
from easyfsl.data_tools import EasySet
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from datasets.cifar import FewShotCIFAR100
from datasets.mini_imagenet import MiniImageNet


def define_outlier_set(dataset_object):
    class OutlierSet(dataset_object):
        def __init__(
            self,
            specs_file: Union[Path, str],
            training=False,
            swaps=None,
            image_size=224,
            **kwargs
        ):
            super().__init__(
                specs_file=specs_file,
                image_size=image_size,
                training=training,
                **kwargs
            )
            self.swapped_labels = list(self.labels)
            self.swaps = swaps
            if swaps is not None:
                self._swap_labels()
            self.outlier_mode = False

            self.features = None
            self.model = None
            self.device = None

        def set_model(self, model, device):
            self.model = model
            self.device = device
            self.compute_features()

        def compute_features(self):
            features_array_list = []
            for imgs, _ in DataLoader(self, batch_size=64):
                imgs = imgs.to(torch.device(self.device))
                features_array_list.append(
                    self.model.backbone(imgs).detach().cpu().numpy()
                )
            self.features = np.concatenate(features_array_list, axis=0)

        def __getitem__(self, item: int):
            img, _ = super().__getitem__(item)
            if self.outlier_mode:
                return (img, self.outlier_labels[item])
            label = self.swapped_labels[item]
            return (img, label)

        def set_swaps(self, swaps: List[Tuple[int, int]] = None, n_outliers=500):
            if swaps is None:
                outlier_indices = np.random.choice(len(self.labels), n_outliers)
                swaps = [
                    (outlier_indices[ind], outlier_indices[ind + 1])
                    for ind in range(0, n_outliers // 2, 2)
                ]
            self.swaps = swaps
            self._swap_labels()
            return self.swaps

        def _swap_labels(self):
            for index1, index2 in self.swaps:
                self.swapped_labels[index1] = index2
                self.swapped_labels[index2] = index1

        def get_outlier_labels(self):
            outlier_labels = [0] * len(self.labels)
            for index, img_label in enumerate(self.labels):
                if self.swapped_labels[index] != img_label:
                    outlier_labels[index] = 1
            return outlier_labels

        def get_swaps(self):
            return self.swaps

        def select_indices_for_label(self, class_index):
            return [i for i, label in enumerate(self.labels) if label == class_index]

        def sample_class_features_with_outliers(
            self,
            class_indices: Optional[List] = None,
            proportion_outliers: float = 0.1,
            limit_num_samples: Optional[int] = None,
            max_num_classes: int = 1,
        ):

            if class_indices is None:
                all_unique_classes = list(set(self.labels))
                if max_num_classes <= len(all_unique_classes) - 1:  # sanity check
                    raise ValueError(
                        "There are not enough classes in the dataset for this experiment"
                    )
                num_classes = np.random.choice(max_num_classes) + 1
                class_indices = np.random.choice(
                    all_unique_classes, num_classes, replace=False
                )

            item_indices = [
                i for i, label in enumerate(self.labels) if label in class_indices
            ]
            other_indices = [
                i for i, label in enumerate(self.labels) if label not in class_indices
            ]

            if limit_num_samples is not None and limit_num_samples < len(item_indices):
                item_indices = list(
                    np.random.choice(item_indices, size=limit_num_samples)
                )

            num_outliers = int(len(item_indices) * proportion_outliers)
            indices_to_swap = np.random.choice(
                range(len(item_indices)), size=num_outliers
            )
            swap_target_images = np.random.choice(other_indices, size=num_outliers)

            outlier_labels = [False] * len(item_indices)

            for swap_index, item_index in enumerate(indices_to_swap):
                item_indices[item_index] = swap_target_images[swap_index]
                outlier_labels[item_index] = True

            return np.take(self.features, item_indices, axis=0), outlier_labels

        def activate_outlier_mode(self):
            self.outlier_labels = self.get_outlier_labels()
            self.outlier_mode = True

        def disable_outlier_mode(self):
            self.outlier_mode = False

    return OutlierSet


OutlierCIFAR = define_outlier_set(FewShotCIFAR100)
OutlierEasySet = define_outlier_set(EasySet)
OutlierMiniImageNet = define_outlier_set(MiniImageNet)
