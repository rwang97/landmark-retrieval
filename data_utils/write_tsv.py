import csv
from pathlib import Path
import numpy as np

input_index_csv = '/datadrive/google-landmark/index_image_to_landmark.csv'
output_index_dir = Path('/datadrive/google-landmark/landmark-retrieval/ArcFace/inference_results/35000/embeds/index')

landmark = {}
landmark_count = set()
with open(input_index_csv, newline='') as csvfile2:
    csv2 = csv.reader(csvfile2)
    for i, row in enumerate(csv2):
        if i > 0:
            img_id = row[0]
            landmark_id = row[1]

            if landmark_id not in landmark:
                landmark[landmark_id] = [output_index_dir / (img_id + '.npy')]
            else:
                landmark[landmark_id].append(output_index_dir / (img_id + '.npy'))
            
            if len(landmark[landmark_id]) >= 20:
                landmark_count.add(landmark_id)
                
landmark = {k: v for k, v in sorted(landmark.items(), key=lambda item: len(item[1]), reverse=True)}

count = 0
filtered_landmark = {}
for key, value in landmark.items():
    if count == 50:
        break
    if len(value) >= 350 and len(value) <= 900:
        filtered_landmark[key] = value
        print(key, len(value))
        count += 1


label_file = "metadata.tsv"
vector_file = "tensor.tsv"
with open(label_file, 'w', encoding='utf8', newline='') as l_file:
    l_tsv_writer = csv.writer(l_file, delimiter='\t', lineterminator='\n')
    l_tsv_writer.writerow(["landmark_id", "id"])

    with open(vector_file, 'w', encoding='utf8', newline='') as v_file:
        v_tsv_writer = csv.writer(v_file, delimiter='\t', lineterminator='\n')
        for key, value in filtered_landmark.items():
            # if key in ['15690', '41374', '26620', '1466', '62244', '77405', '18375', '46785', '67494', '37244']:
            for i, id in enumerate(value):
                if i <= 50:
                    l_tsv_writer.writerow([key, id.stem])
                    v_tsv_writer.writerow(np.load(id).tolist())
