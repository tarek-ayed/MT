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

        def set_swaps(
            self, swaps: List[Tuple[int, int]] = None, proportion_outliers=0.1
        ):
            # TODO: implement or delete
            pass

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
            # TODO: implement sampling logic here
            return [i for i, label in enumerate(self.labels) if label == class_index]

        def get_class_outlier_labels(self, class_index):
            # TODO: implement sampling logic here
            return [
                self.outlier_labels[i]
                for i, label in enumerate(self.labels)
                if label == class_index
            ]

        def activate_outlier_mode(self):
            self.outlier_labels = self.get_outlier_labels()
            self.outlier_mode = True

        def disable_outlier_mode(self):
            self.outlier_mode = False

    return OutlierSet


OutlierCIFAR = define_outlier_set(FewShotCIFAR100)
OutlierEasySet = define_outlier_set(EasySet)
