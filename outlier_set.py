#%%
from pathlib import Path
from typing import List, Tuple, Union
from easyfsl.data_tools import EasySet
import numpy as np


class OutlierSet(EasySet):
    def __init__(
        self, specs_file: Union[Path, str], image_size=224, training=False, swaps=None
    ):
        super().__init__(specs_file, image_size=image_size, training=training)
        self.swapped_labels = list(self.labels)
        self.swaps = swaps
        if swaps is not None:
            self._set_swaps()

    def __getitem__(self, item: int):
        img, _ = super().__getitem__(item)
        label = self.swapped_labels[item]
        return (img, label)
    
    def set_swaps(self, swaps: List[Tuple[int, int]]=None, n_outliers=500):
        if swaps is not None:
            self.swaps = swaps
            self._set_swaps()
        else:
            outlier_indices = np.random.choice(len(self.labels), n_outliers)
            self.swaps = [(outlier_indices[ind], outlier_indices[ind+1]) for ind in range(0,n_outliers//2, 2)]
            self._set_swaps()
        return self.swaps

    def _set_swaps(self):
        for index1, index2 in self.swaps:
            self.swapped_labels[index1] = index2
            self.swapped_labels[index2] = index1
    
    def get_outlier_labels(self):
        outlier_labels = [0] * len(self.labels)
        for index, img_label in enumerate(self.labels):
            if self.swaps[img_label] != img_label:
                outlier_labels[index] = 1
        return(outlier_labels)
    
    def get_swaps(self):
        return(self.swaps)
