import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import *
import misc.utils as utils
import subprocess
from .CaptionModel import CaptionModel
from .AttModel import AttModel


class CBN2D(nn.Module):
    def __init__(self, feat_size, momentum=0.1):
        super(CBN2D, self).__init__()
        self.feat_size = feat_size
        # self.gamma = Parameter(torch.Tensor(self.feat_size))
        # self.beta = Parameter(torch.Tensor(self.feat_size))
        # self.var = Variable(torch.Tensor(self.feat_size).cuda())
        # self.miu = Variable(torch.Tensor(self.feat_size).cuda())
        self.register_buffer('var', torch.ones(self.feat_size))
        self.register_buffer('miu', torch.zeros(self.feat_size))
        self.eps = 1e-9
        self.momentum = momentum
        # self.reset_param()

    def forward(self, x, gatta=None):
        assert(x.dim() == 4)

        if not gatta is None:
            # print(gatta.size())
            gamma = gatta.narrow(1, 0, self.feat_size)
            beta = gatta.narrow(1, self.feat_size, self.feat_size)
        else:
            gamma = Variable(torch.zeros(x.size(0), self.feat_size))
            beta = Variable(torch.zeros(x.size(0), self.feat_size))
        if x.is_cuda and not gamma.is_cuda:
            gamma = gamma.cuda()
            beta = beta.cuda()
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        x = x.permute(0, 2, 1)
        # x = x.view(-1, self.feat_size)
        tmp_x = x.contiguous().view(-1, self.feat_size).data
        tmp_mean = torch.mean(tmp_x, 0)
        tmp_var = torch.var(tmp_x, 0)
        gamma = gamma.unsqueeze(1).expand(x.size())
        beta = beta.unsqueeze(1).expand(x.size())

        out = (x - Variable(self.miu)) / torch.sqrt(Variable(self.var) +
                                                    self.eps) * (gamma + 1) + (beta)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(x_size)

        self.miu = self.momentum * self.miu + \
            (1 - self.momentum) * tmp_mean

        self.var = self.momentum * self.momentum + \
            (1 - self.momentum) * tmp_var
        return out

    # def cuda(self):
    #     # print('Just cuda method CALLED!')
    #     self.var = self.var.cuda()
    #     self.miu = self.miu.cuda()
    #     super(CBN2D, self).cuda()

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
        # self.alpha_beta1 = nn.Sequential(
        #     nn.Linear(opt.rnn_size, opt.rnn_size), nn.ReLU(), nn.Linear(opt.rnn_size, 2))
        # self.alpha_beta2 = nn.Sequential(
        #     nn.Linear(opt.rnn_size, opt.rnn_size), nn.ReLU(), nn.Linear(opt.rnn_size, 2))
        self.alpha_beta1 = nn.Sequential(
            nn.Linear(opt.rnn_size, opt.rnn_size), nn.ReLU(), nn.Linear(opt.rnn_size, 2 * opt.rnn_size))
        self.alpha_beta2 = nn.Sequential(
            nn.Linear(opt.rnn_size, opt.rnn_size), nn.ReLU(), nn.Linear(opt.rnn_size, 2 * opt.rnn_size))

    def forward(self, att_feat, embed_xt=None):
        gatta1 = None
        gatta2 = None
        if not embed_xt is None:
            # print(embed_xt.size())
            gatta1 = self.alpha_beta1(embed_xt)
            gatta2 = self.alpha_beta2(embed_xt)
        res = self.conv1(att_feat)
        x = self.bn1(res, gatta1)
        # print(x.size())
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn2(x, gatta2)
        x = F.relu(x + res)
        return x


class ResSeq(nn.Module):
    def __init__(self, opt):
        super(ResSeq, self).__init__()
        self.reslist = nn.ModuleList([ResBlk(opt)
                                      for i in range(opt.resblock_num)])
        self.resblock_num = opt.resblock_num
        self.pool = nn.MaxPool2d(14)

    def forward(self, att_feat, prev_hidden):
        for i in range(self.resblock_num):
            att_feat = self.reslist[i](att_feat, prev_hidden)
        x = self.pool(att_feat)
        return torch.squeeze(x)


class ResCore(nn.Module):
    def __init__(self, opt):
        super(ResCore, self).__init__()
        self.lstm = ResLSTM(opt, opt.input_encoding_size + opt.rnn_size)
        self.resblocks = ResSeq(opt)
        # self.prev_out = None

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        # xt: batch * 512
        # fc_feats batch*512
        # att_feats batch*512
        # p_att_feats batch*512
        # print('step')
        conv_x = att_feats.permute(0, 3, 1, 2)
        # print(conv_x.size())
        # print(state[0].size())
        conved_input = self.resblocks(conv_x, state[0][0])
        lstm_input = torch.cat((conved_input, xt), 1)
        # print(state[0].size())
        out, state = self.lstm(lstm_input, state)
        # print(state.size())
        # self.prev_out = out.squeeze()
        # result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        # print(result.stdout.decode('utf-8'))
        # state = (out, hidden)
        # if state[0].dim() != 3:
        #     state[0] = state[0].unsqueeze(0)
        #     state[1] = state[1].unsqueeze(0)
        return out, state


class ResLSTM(nn.Module):
    def __init__(self, opt, input_size):
        super(ResLSTM, self).__init__()
        self.rnn_size = opt.rnn_size
        self.input_size = input_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.i2h = nn.Linear(self.input_size, 4 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)

    def forward(self, input_var, state):
        input_sum = self.i2h(input_var) + self.h2h(state[0][-1])
        # print(input_sum.size())
        sigmoid_chunk = input_sum.narrow(1, 0, self.rnn_size * 3)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = input_sum.narrow(
            1, self.rnn_size * 3, self.rnn_size)
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class ResModel(AttModel):
    def __init__(self, opt):
        super(ResModel, self).__init__(opt)
        self.core = ResCore(opt)

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #     # print(weight)
    #     return (Variable(weight.new(bsz, self.rnn_size).zero_()),
    #             Variable(weight.new(bsz, self.rnn_size).zero_()))
