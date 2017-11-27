import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
from .AttModel import AttModel


class CBN2D(nn.Module):
    def __init__(self, feat_size, momentum=0.1):
        super(CBN2D, self).__init__()
        self.feat_size = feat_size
        # self.gamma = Parameter(torch.Tensor(self.feat_size))
        # self.beta = Parameter(torch.Tensor(self.feat_size))
        self.var = Variable(torch.Tensor(self.feat_size))
        self.miu = Variable(torch.Tensor(self.feat_size))
        # self.register_buffer('var',torch.ones(self.feat_size))
        # self.register_buffer('miu',torch.zeros(self.feat_size))
        self.eps = 1e-9
        self.momentum = momentum
        # self.reset_param()

    def forward(self, x, gatta=None):
        assert(x.dim() == 4)

        if gatta:
            gamma = gatta[:self.feat_size]
            beta = gatta[self.feat_size:]
        else:
            gamma = Variable(torch.zeros(self.feat_size))
            beta = Variable(torch.zeros(self.feat_size))
        if x.is_cuda and not gamma.is_cuda:
            gamma = gamma.cuda()
            beta = beta.cuda()

        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        x = x.permute(0, 2, 1)
        # x = x.view(-1, self.feat_size)
        tmp_x = x.contiguous().view(-1, self.feat_size)
        tmp_mean = torch.mean(tmp_x, 0)
        tmp_var = torch.var(tmp_x, 0)
        # print(type(x.data))
        out = (x - self.miu) / torch.sqrt(self.var +
                                          self.eps) * (gamma + 1) + (beta)
        out = out.permute(0, 2, 1).contiguous()
        out.view(x_size)

        self.miu = self.momentum * self.miu + \
            (1 - self.momentum) * tmp_mean

        self.var = self.momentum * self.momentum + \
            (1 - self.momentum) * tmp_var
        return out

    def cuda(self):
        # print('Just cuda method CALLED!')
        self.var = self.var.cuda()
        self.miu = self.miu.cuda()
        super(CBN2D, self).cuda()
        

    # def reset_param(self):
    #    self.miu.data.zero_()
    #    self.var.data.fill_(1)


class ResBlk(nn.Module):

    def __init__(self, opt):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(opt.rnn_size, opt.rnn_size, 1), nn.ReLU())
        self.conv2 = nn.Conv2d(opt.rnn_size, opt.rnn_size, 3, padding=1)
        self.bn1 = CBN2D(opt.rnn_size)
        self.conv3 = nn.Conv2d(opt.rnn_size, opt.rnn_size, 3, padding=1)
        self.bn2 = CBN2D(opt.rnn_size)
        self.alpha_beta1 = nn.Sequential(
            nn.Linear(opt.rnn_size, opt.rnn_size * 2), nn.ReLU())
        self.alpha_beta2 = nn.Sequential(
            nn.Linear(opt.rnn_size, opt.rnn_size * 2), nn.ReLU())

    def forward(self, att_feat, embed_xt=None):
        gatta1 = None
        gatta2 = None
        if embed_xt:
            gatta1 = self.alpha_beta1(embed_xt)
            gatta2 = self.alpha_beta2(embed_xt)
        res = self.conv1(att_feat)
        x = self.bn1(res, gatta1)
        # print(x.size())
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x + res)
        return x


class ResSeq(nn.Module):
    def __init__(self, opt):
        super(ResSeq, self).__init__()
        self.reslist = nn.ModuleList([ResBlk(opt)
                                      for i in range(opt.resblock_num)])
        self.resblock_num = opt.resblock_num
        self.pool = nn.MaxPool2d(14)

    def forward(self, att_feat, embed_xt):
        for i in range(self.resblock_num):
            x = self.reslist[i](att_feat, embed_xt)
        x = self.pool(x)
        return torch.squeeze(x)


class ResCore(nn.Module):
    def __init__(self, opt):
        super(ResCore, self).__init__()
        self.lstm = nn.LSTMCell(
            opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.resblocks = ResSeq(opt)
        self.prev_out = None

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        # xt: batch * 512
        # fc_feats batch*512
        # att_feats batch*512
        # p_att_feats batch*512

        conv_x = att_feats.permute(0, 3, 1, 2)
        # print(conv_x.size())
        lstm_input = self.resblocks(conv_x, self.prev_out)
        out, state = self.lstm(lstm_input, state)
        self.prev_out = out
        return out, state


class ResModel(AttModel):
    def __init__(self, opt):
        super(ResModel, self).__init__(opt)
        self.core = ResCore(opt)
