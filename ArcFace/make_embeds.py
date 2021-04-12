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
import cv2
import shutil
np.set_printoptions(threshold=sys.maxsize)

def process_img(img, size):
    height, width, channels = img.shape
    if width == height:
        new_img = cv2.resize(img, (size, size))
    else:
        ratio = float(size)/max(height, width)
        new_height = int(height*ratio)
        new_width = int(width*ratio)

        new_img = cv2.resize(img, (new_width, new_height))

        delta_w = size - new_width
        delta_h = size - new_height
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        new_img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    return np.transpose(new_img, (2, 0, 1))

# multi processing version
def multi_parse(output_dir, encoder_path, img):
    saved_img_f = output_dir / img.stem
    if os.path.exists(str(saved_img_f) + '.npy'):
        print("img {} embedding exists".format(img.stem))
        return
    # Load the models one by one.
    print("============ running encoder on {} ============".format(img.stem))

    # https://stackoverflow.com/questions/50412477/python-multiprocessing-grab-free-gpu
    if torch.cuda.is_available():
        cpu_name = multiprocessing.current_process().name
        cpu_id = int(cpu_name[cpu_name.find('-') + 1:])
        gpu_id = cpu_id % torch.cuda.device_count()
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    encoder.load_model(encoder_path, multi_gpu=False, device=device)
    input_img = process_img(cv2.imread(str(img)), 224)/255. 
    embedding = encoder.embed_img(input_img)

    torch.cuda.empty_cache()
    
    if os.path.exists(str(saved_img_f) + '.npy'):
        print("img {} embedding exists".format(img.stem))
        return
    np.save(saved_img_f, embedding)

def run_model(img_list, output_dir, encoder_path):
    pool = multiprocessing.Pool(4)
    func = partial(multi_parse, output_dir, encoder_path)
    pool.map(func, img_list)
    pool.close()
    pool.join()

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="train_ckpts/arcface_2_backups/ckpt/arcface_2_16250.pt",
                        help="Path to a saved encoder")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--input_csv", type=Path, default='/datadrive/google-landmark/retrieval_solution_v2.1.csv')
    parser.add_argument("--input_index_csv", type=Path, default='/datadrive/google-landmark/index_image_to_landmark.csv')
    parser.add_argument("--output_test_dir", type=Path, default='inference_results/16250/embeds/test')
    parser.add_argument("--output_index_dir", type=Path, default='inference_results/16250/embeds/index')
    parser.add_argument("--output_small_index_dir", type=Path, default='inference_results/16250/embeds/index_small')
    parser.add_argument("--use_large_index_set", type=bool, default=True)

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

    index_list = []
    small_index_list = []
    test_list = []
    with open(args.input_csv, newline='') as csvfile:
        csv1 = csv.reader(csvfile)
        for i, row in enumerate(csv1):
            if i > 0 and row[2] != "Ignored":
                img_id = row[0]
                test_path = Path("/datadrive/google-landmark/test") / img_id[0] / img_id[1] / img_id[2] / (img_id + ".jpg")
                test_list.append(test_path)

                index_img_ids = row[1].split()
                index_img_paths = [Path("/datadrive/google-landmark/index") / index_img_id[0] / index_img_id[1] / index_img_id[2] / (index_img_id + ".jpg") for index_img_id in index_img_ids]
                index_list.extend(index_img_paths)
                small_index_list.extend(index_img_ids)
    
    small_index_list = set(small_index_list)
    index_list = set(index_list)
    if args.use_large_index_set:
        with open(args.input_index_csv, newline='') as csvfile2:
            csv2 = csv.reader(csvfile2)
            for i, row in enumerate(csv2):
                if i > 0 and i % 90 == 0:
                    img_id = row[0]
                    index_img_path = Path("/datadrive/google-landmark/index") / img_id[0] / img_id[1] / img_id[2] / (img_id + ".jpg")
                    index_list.add(index_img_path)

    args.output_test_dir.mkdir(exist_ok=True, parents=True)
    run_model(test_list, args.output_test_dir, args.enc_model_fpath)

    args.output_index_dir.mkdir(exist_ok=True, parents=True)
    run_model(index_list, args.output_index_dir, args.enc_model_fpath)

    if args.use_large_index_set:
        args.output_small_index_dir.mkdir(exist_ok=True, parents=True)
        for index_img in args.output_index_dir.iterdir():
            if index_img.stem in small_index_list:
                shutil.copy(index_img, args.output_small_index_dir / index_img.name)

    print("number of images in test set: {}".format(len(test_list)))
    print("number of images in index set: {}".format(len(index_list)))
    print("number of images in small index set: {}".format(len(small_index_list)))
