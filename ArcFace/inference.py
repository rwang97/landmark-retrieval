from encoder import inference as encoder
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import glob
import gzip
import numpy as np
import argparse
import torch
import sys
import pickle
import multiprocessing
from functools import partial

# np.set_printoptions(threshold=sys.maxsize)

def get_gpu_info():
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

def get_cos_similarity(player_embeds_dir):

    player_embeds = []
    players = {}
    index = 0
    for player_embed_f in player_embeds_dir.iterdir():
        player_name = player_embed_f.stem
        # print(player_name, player_elo_dict[player_name])
        # print(player_name)

        players[player_name] = index
        index += 1
        player_embed = np.load(player_embed_f)

        array_sum = np.sum(player_embed)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            print(player_embed)

        player_embeds.append(player_embed)

    player_embeds = np.array(player_embeds)

    similarity = cosine_similarity(player_embeds)

    return players, player_embeds, similarity

def multi_test(player_embeds, players, encoder_path, num_test_games, test_player_dir):

    player_name = test_player_dir.stem
    if player_name not in players.keys():
        print("{} not in training".format(player_name))
        return

    # https://stackoverflow.com/questions/50412477/python-multiprocessing-grab-free-gpu
    cpu_name = multiprocessing.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:])
    gpu_id = cpu_id % torch.cuda.device_count()
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    encoder.load_model(encoder_path, multi_gpu=False, device=device)

    player_games = [np.load(gzip.GzipFile(f, "r")) for f in test_player_dir.iterdir()]
    accuracy = -1
    if len(player_games) == 0:
        print("{}, empty folder, missing test data...".format(player_name))
    else:
        game_embeds = encoder.embed_games(player_games, num_test_games=num_test_games, drop_last=True)
        torch.cuda.empty_cache()
        sims = np.inner(game_embeds, player_embeds)
        preds = np.argmax(sims, axis=1)
        player_index = players[player_name]
        preds_correct = len(preds[preds==player_index])
        accuracy = preds_correct / len(preds)
        print("Player: {}, Accuracy: {}, Number of Games: {}".format(player_name, accuracy, len(player_games)))
    
    return (player_name, accuracy)

def run_model(player_embeds, players, encoder_path, num_test_games, verification_data_dir):
    pool = multiprocessing.Pool(4)
    func = partial(multi_test, player_embeds, players, encoder_path, num_test_games)
    data = pool.map(func, verification_data_dir.iterdir())
    pool.close()
    pool.join()

    return dict(data)

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
    parser.add_argument("--player_embeds_dir", type=Path, 
                        # default="inference_results/player_embeds/embeds_bidirectional_data10000_train")
                        # default="inference_results/player_embeds/embeds_transformer_data10000_train")
                        # default="inference_results/player_embeds/embeds_transformer_data10000_validate_train")
                        # default="inference_results/player_embeds/embeds_bidirectional_data10000_validate_train")
                        default="inference_results/player_embeds/embeds_transformer_data10000_validate_train_opening_23000")
    parser.add_argument("--verification_data_dir", type=Path, default="/datadrive/new_encoder_data/processed/explore/10000/test")
    parser.add_argument("--num_test_games", type=int, default=100)
    # parser.add_argument("--save_name", type=str, default="inference_results/pickles/player_acc_bidirectional_100games")
    # parser.add_argument("--save_name", type=str, default="inference_results/pickles/player_acc_transformer_1game")
    # parser.add_argument("--save_name", type=str, default="inference_results/pickles/player_acc_transformer_validate_100game")
    # parser.add_argument("--save_name", type=str, default="inference_results/pickles/player_acc_bidirectional_validate_100game")
    parser.add_argument("--save_name", type=str, default="inference_results/pickles/player_acc_transformer_validate_opening_100game")

    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    
    args = parser.parse_args()
    
    get_gpu_info()

    players, player_embeds, similarity = get_cos_similarity(args.player_embeds_dir)
    print("================= Player Verification Accuracy =================")

    player_acc = run_model(player_embeds, players, args.enc_model_fpath, args.num_test_games, args.verification_data_dir)    
    print("================= Finished Running =================")

    player_acc = {k: v for k, v in sorted(player_acc.items(), key=lambda item: item[1], reverse=True)}
    for key, value in player_acc.items():
        print(key, value)

    pickle.dump(player_acc, open(args.save_name + ".pkl", "wb"))