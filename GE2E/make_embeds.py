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

# multi processing version
def multi_parse(output_dir, encoder_path, player_dir):
    saved_player_f = output_dir / player_dir.stem
    if os.path.exists(str(saved_player_f) + '.npy'):
        print("player {} embedding exists".format(player_dir.stem))
        return
    # Load the models one by one.
    print("============ running encoder on {} ============".format(player_dir.stem))

    # https://stackoverflow.com/questions/50412477/python-multiprocessing-grab-free-gpu
    cpu_name = multiprocessing.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:])
    gpu_id = cpu_id % torch.cuda.device_count()
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    encoder.load_model(encoder_path, multi_gpu=False, device=device)

    player_games = [np.load(gzip.GzipFile(f, "r")) for f in player_dir.iterdir()]
    player_embed = encoder.embed_player(player_games)

    torch.cuda.empty_cache()
    
    if os.path.exists(str(saved_player_f) + '.npy'):
        print("player {} embedding exists".format(player_dir.stem))
        return
    np.save(saved_player_f, player_embed)

def run_model(player_dirs, output_dir, encoder_path):
    pool = multiprocessing.Pool(4)
    func = partial(multi_parse, output_dir, encoder_path)
    pool.map(func, player_dirs)
    pool.close()
    pool.join()

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        # default="train_ckpts/encoder_saved_models/validate_train_bidirection_data10000_backups/ckpt/validate_train_bidirection_data10000_21400.pt",
                        # default="train_ckpts/encoder_saved_models/transformer_data10000_backups/ckpt/transformer_data10000_23600.pt",
                        # default="train_ckpts/encoder_saved_models/transformer_data10000_validate_backups/ckpt/transformer_data10000_validate_22400.pt",
                        # default="train_ckpts/encoder_saved_models/bidirectional_data10000_validate_backups/ckpt/bidirectional_data10000_validate_24400.pt",
                        default="train_ckpts/encoder_saved_models/transformer_data10000_validate_opening_backups/ckpt/transformer_data10000_validate_opening_23000.pt",
                        help="Path to a saved encoder")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--games_dir", type=Path, default='/datadrive/new_encoder_data/processed/explore/10000/train')
    parser.add_argument("--output_dir", type=Path, 
                        # default='inference_results/player_embeds/embeds_bidirectional_data10000_validate_train_24400')
                        default='inference_results/player_embeds/embeds_transformer_data10000_validate_train_opening_23000')

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

    # handle input and output dir
    args.output_dir.mkdir(exist_ok=True, parents=True)
    player_dirs = [f for f in args.games_dir.glob("*") if f.is_dir() and any(f.iterdir())]
    # player_dirs.reverse()
    run_model(player_dirs, args.output_dir, args.enc_model_fpath)        

    # Embeddings are L2-normalized (this isn't important here, but if you want to make your own 
    # embeddings it will be).
    # embed /= np.linalg.norm(embed)