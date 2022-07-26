try:
    from .CA_model import ContextualAttention_Enhance
    from .SK_model import SKUnit
except:
    from model.CA_model import ContextualAttention_Enhance
    from model.SK_model import SKUnit

import torch
import torch.nn as nn
import numpy as np

class merge_block(nn.Module):
    def __init__(self,in_channels,out_channels,vector_length=32,use_multiple_size=False,use_topk=False):
        super(merge_block,self).__init__()
        self.SKUnit = SKUnit(in_features=in_channels,out_features=out_channels,M=2,G=8,r=2)
        self.CAUnit = ContextualAttention_Enhance(in_channels=in_channels,use_multiple_size=use_multiple_size)
        self.fc1 = nn.Linear(in_features=in_channels,out_features=vector_length)
        self.att_CA = nn.Linear(in_features=vector_length,out_features=out_channels)
        self.att_SK = nn.Linear(in_features=vector_length,out_features=out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, region=None, ca_forward_type="forward",
                flows=None, exact=False, ws=29, wt=0, k=100, sb=None):
        out1 = self.SKUnit(x)[:,None]
        if ca_forward_type in ["forward","default"]:
            out2 = self.CAUnit(x,region,flows,exact)[:,None]
        elif ca_forward_type == "dnls":
            out2 = self.CAUnit.dnls_forward(x,region,flows,exact)[:,None]
        elif ca_forward_type == "dnls_k":
            out2 = self.CAUnit.dnls_k_forward(x,region,flows,exact,ws,wt,k,sb)[:,None]
        else:
            raise ValueError(f"Uknown CrossAttn forward type [{ca_forward_type}]")
        out = torch.cat((out2,out1),dim=1)
        U = torch.sum(out,dim=1)
        attention_vector = U.mean(-1).mean(-1)
        attention_vector = self.fc1(attention_vector)
        attention_vector_CA = self.att_CA(attention_vector)[:,None]#.unsqueeze_(dim=1)
        attention_vector_SK = self.att_SK(attention_vector)[:,None]#.unsqueeze_(dim=1)
        vector = torch.cat((attention_vector_CA,attention_vector_SK),dim=1)
        vector = self.softmax(vector)[...,None,None]#.unsqueeze(-1).unsqueeze(-1)
        out = (out*vector).sum(dim=1)
        return out
