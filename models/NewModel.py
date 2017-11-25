import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
from .AttModel import AttModel

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


class ResCaption(CaptionModel):
    def __init__(self, opt):
        super(ResCore, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))

    
    def forward(self, fc_feats, att_feats, seq):

        
    
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(
            xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(
            *(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(
            *(att_feats.size()[:-1] + (self.att_hid_size,)))

        assert beam_size <= self.vocab_size + \
            1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k +
                                    1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k + 1].expand(
                *((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(
                *((beam_size,) + p_att_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(
                    xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(
                state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, opt=opt)
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(
            *(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(
            *(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(
                        torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                # gather the logprobs at sampled positions
                sampleLogprobs = logprobs.gather(
                    1, Variable(it, requires_grad=False))
                # and flatten indices for downstream processing
                it = it.view(-1).long()

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step

                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(
                xt, fc_feats, att_feats, p_att_feats, state)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
