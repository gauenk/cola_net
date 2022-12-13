
# -- python --
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from colanet.utils.misc import rslice
from easydict import EasyDict as edict

# -- modules --
from . import shared_mods
from . import inds_buffer
from ..utils.timer import ExpTimer,ExpTimerList,AggTimer

# -- modules --
from colanet.utils import clean_code
from colanet.utils.config_blocks import config_to_list
from .misc_blocks import default_conv,ResBlock,MeanShift
from .merge_unit import merge_block

@clean_code.add_methods_from(shared_mods)
@clean_code.add_methods_from(inds_buffer)
class RR(nn.Module):

    def __init__(self, args, search_cfg,
                 conv=default_conv):
        super(RR, self).__init__()

        # -- init --
        n_resblocks = 16  # args.n_resblocks
        n_feats = 64  # args.n_feats
        kernel_size = 3
        self.return_inds = args.arch_return_inds
        self.n_resblocks = n_resblocks
        if search_cfg is None: search_cfg = {}

        # -- static --
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        # -- init --
        msa = CES(in_channels=n_feats,num=args.stages,
                  return_inds=args.arch_return_inds,
                  attn_timer=args.attn_timer,search_cfg=search_cfg)
        self.msa = msa
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = []
        for _ in range(n_resblocks // 2):
            m_body.append(ResBlock(conv, n_feats, kernel_size,
                                   nn.PReLU(), res_scale=args.res_scale))
        m_body.append(msa)
        for _ in range(n_resblocks // 2):
            m_body.append(ResBlock(conv, n_feats, kernel_size,
                                   nn.PReLU(), res_scale=args.res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        # -- arch --
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        # -- inds buffer --
        self.use_inds_buffer = self.return_inds
        self.inds_buffer = []

    @property
    def times(self):
        return self.msa.times

    def reset_times(self):
        self.msa.reset_times()

    def forward(self, x, flows=None, inds=None):
        res = x
        res = self.head(res)
        for _name,layer in self.body.named_children():
            if int(_name) == 8: res,inds = layer(res,flows,inds)
            else: res = layer(res)
        res = self.tail(res)
        self.inds_buffer = inds
        # self.update_inds_buffer(inds)
        return x+res

@clean_code.add_methods_from(inds_buffer)
class CES(nn.Module):

    def __init__(self, in_channels, num=6,
                 return_inds=False, attn_timer=False,
                 search_cfg=None):
        super(CES,self).__init__()
        RBS1 = []
        for _ in range(num//2):
            RBS1.append(ResBlock(default_conv, n_feats=in_channels,
                                 kernel_size=3, act=nn.PReLU(), res_scale=1))
        RBS2 = []
        for _ in range(num//2):
            RBS2.append(ResBlock(default_conv, n_feats=in_channels,
                                 kernel_size=3, act=nn.PReLU(), res_scale=1))
        self.RBS1 = nn.Sequential(*RBS1)
        self.RBS2 = nn.Sequential(*RBS2)

        search_cfg_l = config_to_list(search_cfg)
        kwargs = {"in_channels":in_channels,
                  "out_channels":in_channels}
        # print(search_cfg_l[0])
        # print(search_cfg_l[1])
        # print(search_cfg_l[2])
        kwargs['search_cfg'] = search_cfg_l[0]
        self.c1 = merge_block(**kwargs)
        kwargs['search_cfg'] = search_cfg_l[1]
        self.c2 = merge_block(**kwargs)
        kwargs['search_cfg'] = search_cfg_l[2]
        self.c3 = merge_block(**kwargs)
        self.return_inds = return_inds
        self.use_inds_buffer = return_inds
        self.use_timer = attn_timer
        self.times = ExpTimerList(attn_timer)

    def reset_times(self):
        self._reset_times()
        for i in range(3):
            layer_i = getattr(self,'c%d'%(i+1))
            layer_i._reset_times()

    def update_ca_times(self):
        for i in range(3):
            layer_i = getattr(self,'c%d'%(i+1))
            self.update_times(layer_i.times)
            layer_i._reset_times()

    def forward(self, x, flows=None, inds=None):
        self.clear_inds_buffer()
        if not(inds is None):
            inds0,inds1,inds2 = inds[0],inds[1],inds[2]
            out,inds0 = self.c1(x,flows,inds0)
            out = self.RBS1(out)
            out,inds1 = self.c2(out,flows,inds1)
            out = self.RBS2(out)
            out,inds2 = self.c3(out,flows,inds2)
        else:
            out,inds0 = self.c1(x,flows,inds)
            out = self.RBS1(out)
            out,inds1 = self.c2(out,flows,inds0)
            out = self.RBS2(out)
            out,inds2 = self.c3(out,flows,inds1)
        inds = self.format_inds(inds0,inds1,inds2)
        self.update_ca_times()
        return out,inds

