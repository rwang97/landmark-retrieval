import csv 
import os
from pathlib import Path
import shutil
import multiprocessing
from functools import partial

def multi_process(labels, output_root_dir, label):
    id_list = labels[label]
    for img_id in id_list:
        path = Path("train") / img_id[0] / img_id[1] / img_id[2] / (img_id + ".jpg")
        output_dir = output_root_dir / label
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / (img_id + ".jpg")
        print("copying {} to {}".format(path, output_path))
        os.rename(path, output_path)
        

output_root_dir = Path("train_clean_processed")
labels = {}
with open('train_clean.csv', newline='') as csvfile:
  csv = csv.reader(csvfile)
  for i, row in enumerate(csv):
      if i > 0:
        label = row[0]
        id_list = row[1].split()
        labels[label] = id_list

print("finish constructing labels")

pool = multiprocessing.Pool(10)
func = partial(multi_process, labels, output_root_dir)
data = pool.map(func, labels.keys())
pool.close()
pool.join()