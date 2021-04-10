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

def process_img(img, size):
    height, width, channels = img.shape
    if width == height:
        return np.transpose(cv2.resize(img, (size, size)), (2,0,1))

    if width > height:
        new_size = width
    else:
        new_size = height

    new_x = (new_size - width) // 2
    new_y = (new_size - height) // 2
    new_img = cv2.copyMakeBorder(img, new_y, new_y+new_size, new_x, new_x+new_size, cv2.BORDER_CONSTANT, value=[0,0,0])
    return np.transpose(cv2.resize(new_img, (size, size)), (2,0,1))

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
    input_img = process_img(cv2.imread(str(img)), 112)/255. 
    embedding = encoder.embed_img(input_img)

    torch.cuda.empty_cache()
    
    if os.path.exists(str(saved_img_f) + '.npy'):
        print("img {} embedding exists".format(img.stem))
        return
    np.save(saved_img_f, embedding)

def run_model(img_list, output_dir, encoder_path):
    pool = multiprocessing.Pool(3)
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
                        default="train_ckpts/ge2e_1_backups/ckpt/ge2e_1_30200.pt",
                        help="Path to a saved encoder")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--input_csv", type=Path, default='/datadrive/google-landmark/retrieval_solution_v2.1.csv')
    parser.add_argument("--output_test_dir", type=Path, default='inference_results/embeds/test')
    parser.add_argument("--output_index_dir", type=Path, default='inference_results/embeds/index')

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

    count = 0
    index_list = []
    test_list = []
    with open(args.input_csv, newline='') as csvfile:
        csv = csv.reader(csvfile)
        for i, row in enumerate(csv):
            if i > 0 and row[2] != "Ignored":
                img_id = row[0]
                test_path = Path("/datadrive/google-landmark/test") / img_id[0] / img_id[1] / img_id[2] / (img_id + ".jpg")
                test_list.append(test_path)

                index_img_ids = row[1].split()
                index_img_paths = [Path("/datadrive/google-landmark/index") / index_img_id[0] / index_img_id[1] / index_img_id[2] / (index_img_id + ".jpg") for index_img_id in index_img_ids]
                index_list.extend(index_img_paths)

    
    # handle input and output dir
    # args.output_test_dir.mkdir(exist_ok=True, parents=True)
    # run_model(test_list, args.output_test_dir, args.enc_model_fpath)     

    args.output_index_dir.mkdir(exist_ok=True, parents=True)
    run_model(index_list, args.output_index_dir, args.enc_model_fpath)