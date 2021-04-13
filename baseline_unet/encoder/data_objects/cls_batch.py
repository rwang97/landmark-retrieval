import numpy as np
from typing import List
from encoder.data_objects.cls import Cls

class ClsBatch:
    def __init__(self, classes: List[Cls], img_per_cls: int):
        self.classes = classes
        self.samples = [np.array(c.random_sample(img_per_cls)) for c in classes]
        self.data = np.stack(self.samples, axis=0)
        n_cls, n_img, feature_shape = self.data.shape[0], self.data.shape[1], self.data.shape[2:]
        self.data = np.reshape(self.data, (n_cls*n_img, *feature_shape))
        self.labels = np.array([int(c.name) for c in classes for x in range(img_per_cls)])
