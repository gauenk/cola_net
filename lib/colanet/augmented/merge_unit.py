
# -- torch --
import torch
import torch.nn as nn
import numpy as np
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils import fwd_4dim

# -- modules --
from .ca_module import ContextualAttention_Enhance
from .sk_conv import SKUnit

class merge_block(nn.Module):
    def __init__(self,search_cfg,in_channels,out_channels,vector_length=32,
                 use_multiple_size=False,use_topk=False):
        super(merge_block,self).__init__()
        if search_cfg is None: search_cfg = {}
        self.SKUnit = SKUnit(in_features=in_channels,
                             out_features=out_channels,M=2,G=8,r=2)
        self.CAUnit = ContextualAttention_Enhance(search_cfg,in_channels=in_channels,
                                                  use_multiple_size=use_multiple_size)
        self.fc1 = nn.Linear(in_features=in_channels,out_features=vector_length)
        self.att_CA = nn.Linear(in_features=vector_length,out_features=out_channels)
        self.att_SK = nn.Linear(in_features=vector_length,out_features=out_channels)
        self.softmax = nn.Softmax(dim=1)

    @property
    def times(self):
        return self.CAUnit.times

    def _reset_times(self):
        self.CAUnit._reset_times()

    def forward(self, x, flows, state, batchsize):
        out1 = self.SKUnit(x)
        out2 = self.CAUnit(x,flows,state,batchsize)
        out = torch.cat((out2[:,None],out1[:,None]),dim=1)
        U = torch.sum(out,dim=1)
        attention_vector = U.mean(-1).mean(-1)
        attention_vector = self.fc1(attention_vector)
        attention_vector_CA = self.att_CA(attention_vector)[:,None]
        attention_vector_SK = self.att_SK(attention_vector)[:,None]
        vector = torch.cat((attention_vector_CA,attention_vector_SK),dim=1)
        vector = self.softmax(vector)[...,None,None]
        out = (out*vector).sum(dim=1)
        return out
