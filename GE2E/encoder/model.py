from encoder.params_model import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
import collections
from torch import nn
import numpy as np
import torch

class SELayer(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/23839e07525f9f5d39982140fccc8b925fe4dee9/senet/se_module.py
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CenteredBatchNorm2d(torch.nn.BatchNorm2d):
    """Only apply bias, no scale like:
        tf.layers.batch_normalization(
            center=True, scale=False,
            )
    """

    def __init__(self, channels):
        super().__init__(channels, affine = True, eps=1e-5, momentum=0.01)
        # #self.weight = 1 by default
        # self.weight.requires_grad = False

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        layers = [
            ('conv2d', torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride = 1,
                        padding = kernel_size // 2,
                        bias = bias,
                        )),
            ('norm2d', CenteredBatchNorm2d(out_channels)),
            ('ReLU', torch.nn.ReLU()),
            ]
        self.seq = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        return self.seq(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        layers = [
            ('conv2d_1', torch.nn.Conv2d(
                        channels,
                        channels,
                        3,
                        stride = 1,
                        padding = 1,
                        bias = False,
                        )),
            ('norm2d_1', CenteredBatchNorm2d(channels)),
            ('ReLU', torch.nn.ReLU()),
            ('conv2d_2', torch.nn.Conv2d(
                        channels,
                        channels,
                        3,
                        stride = 1,
                        padding = 1,
                        bias = False,
                        )),
            ('norm2d_2', CenteredBatchNorm2d(channels)),
            ('squeeze_excite', SELayer(channels, 8)),
            ]
        self.seq = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        y = self.seq(x)
        y += x
        y = torch.nn.functional.relu(y, inplace = True)
        return y


class Encoder(nn.Module):

    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        self.num_res_blocks = 6
        
        self.conv_block = ConvBlock(in_channels=3, out_channels=64, kernel_size=3)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=64) for i in range(self.num_res_blocks)]
        )

        self.conv_block2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_block4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3)
        self.final_feature = ConvBlock(in_channels=256, out_channels=model_embedding_size, kernel_size=3)

        self.global_avgpool = nn.AvgPool2d(kernel_size=8)

        self.cnn = nn.Sequential(*[
            self.conv_block,
            self.residual_blocks,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.final_feature,
            self.global_avgpool,
            torch.nn.Flatten()
        ])
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])) 
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.GE2E_softmax_loss

    def GE2E_softmax_loss(self, sim_matrix, cls_per_batch, img_per_cls):
        
        # colored entries in paper
        sim_matrix_correct = torch.cat([sim_matrix[i*img_per_cls:(i+1)*img_per_cls, i:(i+1)] for i in range(cls_per_batch)])
        # softmax loss
        loss = -torch.sum(sim_matrix_correct-torch.log(torch.sum(torch.exp(sim_matrix), axis=1, keepdim=True) + 1e-6)) / (cls_per_batch*img_per_cls)
        return loss

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, images):
        """
        Computes the embeddings of a batch of images.
        
        :param images: batch of images of same duration as a tensor of shape 
        (batch_size, channels, img_size, img_size)
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """

        # (batch_size, n_features)
        embeds_raw = self.cnn(images)

        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (cls_per_batch, 
        img_per_cls, embedding_size)
        :return: the similarity matrix as a tensor of shape (cls_per_batch,
        img_per_cls, cls_per_batch)
        """
        cls_per_batch, img_per_cls = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (img_per_cls - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(cls_per_batch, img_per_cls,
                                 cls_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(cls_per_batch, dtype=np.int)
        for j in range(cls_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (cls_per_batch, 
        img_per_cls, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        cls_per_batch, img_per_cls = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((cls_per_batch * img_per_cls, 
                                         cls_per_batch))
        ground_truth = np.repeat(np.arange(cls_per_batch), img_per_cls)
        # target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        # loss = self.loss_fn(sim_matrix, target)
        loss = self.loss_fn(sim_matrix, cls_per_batch, img_per_cls)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, cls_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer