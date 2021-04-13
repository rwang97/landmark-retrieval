from pathlib import Path
import numpy as np
import argparse
import os
import sys
import multiprocessing
from functools import partial
import csv
from test_utils.metrics import metrics
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
    pool = multiprocessing.Pool(4)
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
    parser.add_argument("--test_embeds_path", type=Path, default='/datadrive/google-landmark/landmark-retrieval/ArcFace/inference_results/20250/embeds/test')
    parser.add_argument("--index_embeds_path", type=Path, default='/datadrive/google-landmark/landmark-retrieval/ArcFace/inference_results/20250/embeds/index')
    parser.add_argument("--prediction_csv", type=Path, default='/datadrive/google-landmark/landmark-retrieval/ArcFace/inference_results/20250/prediction.csv')
    parser.add_argument("--groundtruth_csv", type=str, default='/datadrive/google-landmark/retrieval_solution_v2.1.csv')
    
    args = parser.parse_args()
    
    test_embeds = [(test_embed_path.stem, np.load(test_embed_path)) for test_embed_path in args.test_embeds_path.iterdir()]

    index_embeds_name = []
    index_embeds = []
    for index_embed_path in args.index_embeds_path.iterdir():
        index_embeds_name.append(index_embed_path.stem)
        index_embeds.append(np.load(index_embed_path))

    index_embeds = np.array(index_embeds)
    predictions = get_predictions(test_embeds, index_embeds_name, index_embeds)

    # write to csv file
    with open(args.prediction_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in predictions.items():
            writer.writerow([key, " ".join(value)])

    # calculate metrics
    metrics(args.groundtruth_csv, args.prediction_csv)