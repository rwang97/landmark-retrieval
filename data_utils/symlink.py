import csv 
import os
from pathlib import Path
import shutil
import multiprocessing
from functools import partial

with open('train_clean.csv', newline='') as csvfile:
  csv = csv.reader(csvfile)
  for i, row in enumerate(csv):
      if i > 0:
        label = row[0]
        path = Path("train_clean_processed") / label
        symlink_path = Path("train_clean_processed_symlink") / "{}".format(i - 1)
        print(str(path.absolute()), str(symlink_path.absolute()))
        os.symlink(str(path.absolute()), str(symlink_path.absolute()), target_is_directory=True)
