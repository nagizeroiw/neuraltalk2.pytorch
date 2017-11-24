import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel


class CBN2D(nn.Module):
    def __init__(self, feat_size, momentum=0.1):
        super(CBN, self).__init__()
        self.feat_size = feat_size
        # self.gamma = Parameter(torch.Tensor(self.feat_size))
        # self.beta = Parameter(torch.Tensor(self.feat_size))
        self.var = Parameter(torch.Tensor(self.feat_size))
        self.miu = Parameter(torch.Tensor(self.feat_size))
        self.eps = 1e-9
        self.momentum = momentum
        self.reset_param()

    def forward(self, x, gatta=None):
        assert(x.dim() == 4)
        gamma = None
        beta = None
        if gatta:
            gamma = gatta[:self.feat_size]
            beta = gata[self.feat_size:]
        x_size = x.size()
        x = x.view(-1, self.feat_size)
        tmp_mean = torch.mean(x, 0)
        tmp_var = torch.var(x, 0)
        self.out = (x - self.miu) / torch.sqrt(self.var +
                                               self.eps) * (1 + self.gamma or Variable(torch.zeros(self.feat_size))) + (self.beta or Variable(torch.zeros(self.feat_size)))

        if self.training£º
            self.miu = self.momentum * self.miu + \
                (1 - self.momentum) * tmp_mean
            self.var = self.momentum * self.momentum +
                (1 - self.momentum) * tmp_var

    def reset_param(self):
        self.miu.data.zero_()
        self.var.data.fill_(1)


class ResBlk(nn.Module):

    def __init__(self, opt):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2(opt.att_feat_size, opt.att_feat_size, 1), nn.ReLU())
        self.conv2 = nn.Conv2(
            nn.Conv2(opt.att_feat_size, opt.att_feat_size, 3, padding=1))
        self.bn1 = CBN2D(opt.att_feat_size)
        self.conv3 = nn.Conv2(
            nn.Conv2(opt.att_feat_size, opt.att_feat_size, 3, padding=1))
        self.bn2 = CBN2D(opt.att_feat_size)
        self.alpha_beta1 = nn.Sequential(
            nn.Linear(opt.input_encoding_size, opt.att_feat_size * 2), nn.ReLU())
        self.alpha_beta2 = nn.Sequential(
            nn.Linear(opt.att_feat_size, opt.input_encoding_size * 2), nn.ReLU())

    def forward(self, att_feat, embed_xt=None):
        gatta1 = None
        gatta2 = None
        if embed_xt:
            gatta1 = self.alpha_beta1(embed_xt)
            gatta2 = self.alpha_beta2(embed_xt)
        res = self.conv1(att_feat)
        x = self.bn1(res, gatta1)
        F.relu_(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x + res)
        return x


class ResCore(nn.Module):
    def __init__(self, opt):
        super(ResCore, self).__init__()
        self.rnn_size = opt.rnn_size
        self.input_encoding_size = opt.input_encoding_size


class ResCaption(CaptionModel):
    pass
