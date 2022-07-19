try:
    from . import common
    from .merge_unit import merge_block
except:
    import model.common as common
    from model.merge_unit import merge_block

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from colanet.utils.misc import rslice

class RR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RR, self).__init__()

        n_resblocks = 16  # args.n_resblocks
        n_feats = 64  # args.n_feats
        kernel_size = 3
        self.n_resblocks = n_resblocks

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        msa = CES(in_channels=n_feats,num=args.stages)#blocks=args.blocks)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_body.append(msa)
        for i in range(n_resblocks // 2):
            m_body.append(common.ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        # self.tail = nn.Sequential(*m_tail)

        # msa = CES(in_channels=n_feats,num=args.stages)#blocks=args.blocks)
        # i = 0 # init

        # # -- layer 0 -- "define head module"
        # # m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # layer_i = conv(args.n_colors, n_feats, kernel_size)
        # self.add_module('head%d' % i, layer_i)
        # i+=1

        # # -- layers 1-3 -- "define body module"
        # i = 0 # reset
        # # m_body = [
        # #     common.ResBlock(
        # #         conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale
        # #     ) for _ in range(n_resblocks // 2)
        # # ]
        # for _ in range(n_resblocks // 2):
        #     layer_i = common.ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale)
        #     self.add_module('body%d' % i, layer_i)
        #     i+= 1

        # # -- msa layer --
        # # m_body.append(msa)
        # layer_i = msa
        # self.add_module('body%d' % i, layer_i)
        # i+= 1

        # # -- finish out body --
        # for _ in range(n_resblocks // 2):
        #     # m_body.append(common.ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale))
        #     layer_i = common.ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale)
        #     self.add_module('body%d' % i, layer_i)
        #     i+= 1

        # # -- finish of "body" --
        # # m_body.append(conv(n_feats, n_feats, kernel_size))
        # layer_i = conv(n_feats, n_feats, kernel_size)
        # self.add_module('body%d' % i, layer_i)
        # i+= 1

        # # -- tail! --
        # # m_tail = [conv(n_feats, args.n_colors, kernel_size)]
        # i = 0
        # layer_i = conv(n_feats, args.n_colors, kernel_size)
        # self.add_module('tail%d' % i, layer_i)
        # i+= 1

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, region=None, flows=None):
        # Note: "region" is unused in all of the code base.
        # print(x.shape)
        res = x
        # print("a")
        for name, module in self._modules.items():
            if name == "add_mean": continue
            if name == "body":
                for _name,layer in module.named_children():
                    if int(_name) == 8:
                        res = layer(res,region,flows)
                        # print("->",_name)
                    else:
                        res = layer(res)
                        # print("-->",_name)
            else:
                res = module(res)
        #     print("--->",name)
        # print("b")
        return x+res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
class CES(nn.Module):
    def __init__(self,in_channels,num=6):
        super(CES,self).__init__()
        # print('num_RB:',num)
        RBS1 = [
            common.ResBlock(
                common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
            ) for _ in range(num//2)
        ]
        RBS2 = [
            common.ResBlock(
                common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
            ) for _ in range(num//2)
        ]
        self.RBS1 = nn.Sequential(
            *RBS1
        )
        self.RBS2 = nn.Sequential(
            *RBS2
        )
        self.c1 = merge_block(in_channels = in_channels,out_channels=in_channels)#CE(in_channels=in_channels)
        self.c2 = merge_block(in_channels = in_channels,out_channels=in_channels)#CE(in_channels=in_channels)
        self.c3 = merge_block(in_channels = in_channels,out_channels=in_channels)
        # self.ca_forward_type = "default"#dnls_k"
        # self.ca_forward_type = "dnls"
        self.ca_forward_type = "dnls_k"
        self.exact = False

    def forward(self, x, region=None, flows=None):
        out = self.c1(x,region,self.ca_forward_type,flows,self.exact)
        out = self.RBS1(out)
        out = self.c2(out,region,self.ca_forward_type,flows,self.exact)
        out = self.RBS2(out)
        out = self.c3(out,region,self.ca_forward_type,flows,self.exact)
        # print("[ces] out.shape: ",out.shape)
        return out
