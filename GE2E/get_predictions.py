from encoder import inference as encoder
from pathlib import Path
import numpy as np
import argparse
import torch
import os
import sys
import gzip
import multiprocessing
from functools import partial
import csv
# np.set_printoptions(threshold=sys.maxsize)

# multi processing version
def multi_parse(index_embeds_name, index_embeds, test_embed):
    test_embed_name, test_embedding = test_embed
    test_embedding = np.expand_dims(test_embedding, 0)
    sims = np.inner(test_embedding, index_embeds)
    max_indices = np.argsort(-sims.squeeze(0))[:100]
    max_indices = max_indices.tolist()
    max_indices_names = [index_embeds_name[index] for index in max_indices]

    return (test_embed_name, max_indices_names)

    
def get_predictions(test_embeds, index_embeds_name, index_embeds):
    pool = multiprocessing.Pool(1)
    func = partial(multi_parse, index_embeds_name, index_embeds)
    data = pool.map(func, test_embeds)
    pool.close()
    pool.join()

    return dict(data)

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--test_embeds_path", type=Path, default='/datadrive/google-landmark/landmark-retrieval/GE2E/inference_results/embeds/test')
    parser.add_argument("--index_embeds_path", type=Path, default='/datadrive/google-landmark/landmark-retrieval/GE2E/inference_results/embeds/index')
    parser.add_argument("--prediction_csv", type=Path, default='/datadrive/google-landmark/landmark-retrieval/GE2E/inference_results/prediction.csv')

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" % 
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    torch.multiprocessing.set_start_method('spawn')
    
    test_embeds = [(test_embed_path.stem, np.load(test_embed_path)) for test_embed_path in args.test_embeds_path.iterdir()]

    index_embeds_name = []
    index_embeds = []
    for index_embed_path in args.index_embeds_path.iterdir():
        index_embeds_name.append(index_embed_path.stem)
        index_embeds.append(np.load(index_embed_path))

    index_embeds = np.array(index_embeds)
    predictions = get_predictions(test_embeds, index_embeds_name, index_embeds)

    with open(args.prediction_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in predictions.items():
            writer.writerow([key, " ".join(value)])