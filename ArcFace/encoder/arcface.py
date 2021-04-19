import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# modified from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, m=0.50, device=None):
        super(ArcFace, self).__init__()
        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output = output * self.scale
        # print(output)

        return output