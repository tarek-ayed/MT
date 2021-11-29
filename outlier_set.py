from pathlib import Path
from typing import List, Tuple, Union
from easyfsl.data_tools import EasySet
import numpy as np

from cifar import FewShotCIFAR100


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

        def __getitem__(self, item: int):
            img, _ = super().__getitem__(item)
            if self.outlier_mode:
                return (img, self.outlier_labels[item])
            label = self.swapped_labels[item]
            return (img, label)

        def set_swaps(self, swaps: List[Tuple[int, int]] = None, n_outliers=500):
            if swaps is not None:
                self.swaps = swaps
                self._set_swaps()
            else:
                outlier_indices = np.random.choice(len(self.labels), n_outliers)
                self.swaps = [
                    (outlier_indices[ind], outlier_indices[ind + 1])
                    for ind in range(0, n_outliers // 2, 2)
                ]
                self._set_swaps()
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

        def get_class(self, class_index):
            return [i for i, label in enumerate(self.labels) if label == class_index]

        def get_class_with_outliers(self, class_index=None, proportion_outliers=0.1, limit_num_samples=None):
            
            if class_index is None:
                class_index = np.random.choice(self.labels)

            item_indices = [
                i for i, label in enumerate(self.labels) if label == class_index
            ]
            other_indices = [
                i for i, label in enumerate(self.labels) if label != class_index
            ]
            
            if limit_num_samples is not None and limit_num_samples < len(item_indices):
                item_indices = list(np.random.choice(item_indices, size=limit_num_samples))

            num_outliers = int(len(item_indices) * proportion_outliers)
            indices_to_swap = np.random.choice(range(len(item_indices)), size=num_outliers)
            swap_target_images = np.random.choice(other_indices, size=num_outliers)

            outlier_labels = [False] * len(item_indices)

            for swap_index, item_index in enumerate(indices_to_swap):
                item_indices[item_index] = swap_target_images[swap_index]
                outlier_labels[item_index] = True

            return item_indices, outlier_labels

        def activate_outlier_mode(self):
            self.outlier_labels = self.get_outlier_labels()
            self.outlier_mode = True

        def disable_outlier_mode(self):
            self.outlier_mode = False

    return OutlierSet


OutlierCIFAR = define_outlier_set(FewShotCIFAR100)
OutlierEasySet = define_outlier_set(EasySet)
