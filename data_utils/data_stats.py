import os
from pathlib import Path
import statistics
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker

total_count = 0
less_than_5 = 0
stats = {}
root = Path("../../../google-landmark/train_clean_processed")
num_img_dict = {}

for cls in root.iterdir():
    num_imgs = len(list(cls.iterdir()))
    stats[cls.stem] = num_imgs
    total_count += num_imgs
    
    if num_imgs in num_img_dict:
        num_img_dict[num_imgs] += 1
    else:
        num_img_dict[num_imgs] = 1

num_img_dict = collections.OrderedDict(sorted(num_img_dict.items()))
# keys = list(num_img_dict.keys())
# values = list(num_img_dict.values())

# plt.barh(keys, values, align='center')
# plt.savefig('data.png', dpi=300, transparent=False, bbox_inches='tight')

print("number of images: {}".format(num_img_dict))
print("median num of images: {}".format(statistics.median(list(stats.values()))))
print("average num of images: {}".format(statistics.mean(list(stats.values()))))