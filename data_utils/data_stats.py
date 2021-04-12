import os
from pathlib import Path
import statistics

total_count = 0
less_than_8 = 0
stats = {}
root = Path("../../../google-landmark/train_clean_processed")

for cls in root.iterdir():
    num_imgs = len(list(cls.iterdir()))
    stats[cls.stem] = num_imgs
    total_count += num_imgs
    if num_imgs < 8:
        less_than_8 += 1

print("{} classes have less than 8 images".format(less_than_8))
print("median num of images: {}".format(statistics.median(list(stats.values()))))
print("average num of images: {}".format(statistics.mean(list(stats.values()))))