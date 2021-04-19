import os
from pathlib import Path
import statistics
import collections
import shutil
import csv 

input_csv = '/datadrive/google-landmark/retrieval_solution_v2.1.csv'
output_dir = Path('test')


with open(input_csv, newline='') as csvfile:
    csv1 = csv.reader(csvfile)
    for i, row in enumerate(csv1):
        if i > 0 and row[2] != "Ignored":
            img_id = row[0]
            test_path = Path("/datadrive/google-landmark/test") / img_id[0] / img_id[1] / img_id[2] / (img_id + ".jpg")
            output_test_dir = output_dir / img_id
            output_test_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(test_path, output_test_dir/("test.jpg"))

            index_img_ids = row[1].split()
            index_img_paths = [Path("/datadrive/google-landmark/index") / index_img_id[0] / index_img_id[1] / index_img_id[2] / (index_img_id + ".jpg") for index_img_id in index_img_ids]
            [shutil.copy(index_img_path, output_test_dir / index_img_path.name) for index_img_path in index_img_paths]
