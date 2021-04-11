from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.cls_batch import ClsBatch
from encoder.data_objects.cls import Cls
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os

# TODO: improve with a pool of speakers for data efficiency

class LandmarkDataset(Dataset):
    def __init__(self, datasets_root: Path, img_per_cls: int):
        self.root = datasets_root
        cls_dirs = [f for f in self.root.glob("*") if f.is_dir()]

        if len(cls_dirs) == 0:
            raise Exception("No image class found. Make sure you are pointing to the directory "
                            "containing all preprocessed image class directories.")

        self.classes = [Cls(cls_dir) for cls_dir in cls_dirs]
        self.cls_cycler = RandomCycler(self.classes)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.cls_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class LandmarkDataLoader(DataLoader):
    def __init__(self, dataset, cls_per_batch, img_per_cls, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.img_per_cls = img_per_cls

        super().__init__(
            dataset=dataset, 
            batch_size=cls_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, classes):
        return ClsBatch(classes, self.img_per_cls) 
    