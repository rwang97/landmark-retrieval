from encoder.params_model import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
import collections
from torch import nn
import numpy as np
import torch
import torchvision

class Encoder(nn.Module):

    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        self.cnn = torch.nn.Sequential(*(list(self.resnet101.children())[:-1]))
        self.flatten = torch.nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(2048, num_class)

    def forward(self, images):
        """
        Computes the embeddings of a batch of images.
        
        :param images: batch of images of same duration as a tensor of shape 
        (batch_size, channels, img_size, img_size)
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        embeds_raw = self.cnn(images)
        output = self.dropout(embeds_raw)
        output = torch.flatten(output, 1)
        output = self.linear(output)

        return self.flatten(embeds_raw), output
    