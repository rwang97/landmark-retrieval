from encoder.params_model import *
from encoder.model import Unet
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import numpy as np
import torch
import time
_model = None # type: Unet
_device = None # type: torch.device


def load_model(weights_fpath: Path, multi_gpu=False, device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _device

    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device

    checkpoint = torch.load(weights_fpath, _device)
    _model = Unet(_device, _device)

    if multi_gpu:
        if torch.cuda.device_count() <= 1:
            raise "multi_gpu cannot be enabled"

        _model = torch.nn.DataParallel(_model)
        # load params
        _model.load_state_dict(checkpoint["model_state"])
    else:
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
        state_dict = checkpoint['model_state']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.` because of DataParallel
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        # load params
        _model.load_state_dict(new_state_dict)

    _model = _model.to(_device)
    _model.eval()

    # print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))
    
def is_loaded():
    return _model is not None

def embed_imgs_batch(images):
    """
    :param images: shape (batch_size, 34, 8, 8)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    images = torch.from_numpy(images).float().to(_device)
    with torch.no_grad():
        embed, _ = _model.forward(images)
        embed = embed.detach().cpu().numpy()

    return embed

def embed_img(img, **kwargs):
    """
    Computes an embedding for a single image.
    """
    embeds = embed_imgs_batch(np.expand_dims(img, 0))
    
    raw_embed = np.mean(embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    return embed

def embed_imgs(imgs, **kwargs):
    """
    Computes the embedding for a batch of images.
    """
    embeds = embed_imgs_batch(imgs)
    embeds = embeds / np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    
    return embeds

def embed_cls(imgs:list, **kwargs):
    raw_embed = np.mean([embed_img(img, **kwargs) for img in imgs], axis=0)
    return raw_embed / np.linalg.norm(raw_embed, 2)

def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        ax = plt.gca()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)
    
    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
