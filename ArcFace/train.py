from utils.argutils import print_args
from encoder.train import train
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("-d", "--data_dir", type=Path, default="/home/paddlepaddle123/russell/google-landmark/train_clean_processed_symlink", help=\
        "Path to preprocessed data")
    parser.add_argument("-vd", "--validate_data_dir", type=Path, default="/home/paddlepaddle123/russell/google-landmark/train_clean_processed_symlink", help=\
        "Path to preprocessed data")
    parser.add_argument("-m", "--models_dir", type=Path, default="/home/paddlepaddle123/russell/landmark-retrieval/ArcFace/train_ckpts", help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-v", "--vis_every", type=int, default=200, help= \
        "Number of steps between updates of the loss and the plots.")
    parser.add_argument("-u", "--umap_every", type=int, default=200, help= \
        "Number of steps between updates of the umap projection. Set to 0 to never update the "
        "projections.")
    parser.add_argument("-s", "--save_every", type=int, default=200, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=200, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-ve", "--validate_every", type=int, default=200, help= \
        "Number of steps between validation step.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model.")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")
    parser.add_argument("--port", type=str, default="8870")
    parser.add_argument("--no_visdom", action="store_true", help= \
        "Disable visdom.")
    args = parser.parse_args()
    print_args(args, parser)

    return args

if __name__ == "__main__":
    args = parse_args()

    args.models_dir.mkdir(exist_ok=True, parents=True)
    
    train(**vars(args))
    