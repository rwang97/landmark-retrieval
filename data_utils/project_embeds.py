from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import umap

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255 
np.set_printoptions(threshold=sys.maxsize)

def draw_projections(all_landmark_embeds):
    colors = []
    embeds = all_landmark_embeds[0]
    for i in range(len(all_landmark_embeds)):
        colors.extend([colormap[i] for j in range(len(all_landmark_embeds[i]))])
        if i != 0:
            embeds = np.concatenate([embeds, all_landmark_embeds[i]], axis=0)

    # embeds = np.concatenate((embeds1, embeds2), axis=0)
    reducer = umap.UMAP(metric="cosine")
    projected = reducer.fit_transform(embeds)
    plt.scatter(projected[:, 0], projected[:, 1], c=colors, s=10)
    plt.xticks([])
    plt.yticks([])
    plt.title("UMAP Projection (U-Net)")
    plt.gca().set_aspect("equal", "datalim")
    plt.savefig("projection_unet.png", dpi = 300, transparent = True)
    plt.clf()


def get_projection():
    input_index_csv = '/datadrive/google-landmark/index_image_to_landmark.csv'
    output_index_dir = Path('/datadrive/google-landmark/landmark-retrieval/baseline_unet/inference_results/35000/embeds/index')

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


    count = 0
    all_landmark_embeds = []
    for key, value in filtered_landmark.items():
        landmark_embeds = []
        if count < 10:
        # if key in ['15690', '41374', '26620', '1466', '62244', '77405', '18375', '46785', '67494', '37244']:
            print(key)
            for i, id in enumerate(value):
                if i <= 50:
                    landmark_embeds.append(np.load(id))
            landmark_embeds = np.array(landmark_embeds)
            all_landmark_embeds.append(landmark_embeds)
            count += 1
    
    print(len(all_landmark_embeds))
    draw_projections(all_landmark_embeds)


if __name__ == '__main__':
    get_projection()
