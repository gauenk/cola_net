
# -- python --
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from colanet.utils.misc import rslice
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- modules --
from . import shared_mods
from . import inds_buffer
from . import ca_forward
from ..utils.timer import ExpTimer,ExpTimerList,AggTimer
import colanet.utils.gpu_mem as gpu_mem

# -- modules --
from colanet.utils import clean_code
from colanet.utils.config_blocks import config_to_list
from .misc_blocks import default_conv,ResBlock,MeanShift
from .merge_unit import merge_block

@clean_code.add_methods_from(shared_mods)
@clean_code.add_methods_from(inds_buffer)
@clean_code.add_methods_from(ca_forward)
class RR(nn.Module):

    def __init__(self, args, block_cfgs,
                 conv=default_conv):
        super(RR, self).__init__()

        # -- init --
        n_resblocks = 16  # args.n_resblocks
        n_feats = 64  # args.n_feats
        kernel_size = 3
        self.return_inds = args.arch_return_inds
        self.n_resblocks = n_resblocks
        # if block_cfgs is None: block_cfgs = {}

        # -- static --
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        # -- init attn block --
        msa = CES(in_channels=n_feats,num=args.stages,
                  return_inds=args.arch_return_inds,
                  attn_timer=args.attn_timer,block_cfgs=block_cfgs)
        self.msa = msa

        # -- create network --
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

    def forward(self, vid, flows=None, state=None):
        ndim = vid.ndim
        if vid.ndim == 4:
            vid = vid[None,:]
        B = vid.shape[0]
        vid = rearrange(vid,'b t c h w -> (b t) c h w')
        res = vid
        res = self.head(vid)
        for _name,layer in self.body.named_children():
            if int(_name) == 8: res = layer(res,flows,state,B)
            else: res = layer(res)
        res = self.tail(res)
        # self.inds_buffer = inds
        # self.update_inds_buffer(inds)
        vid = vid + res
        vid = rearrange(vid,'(b t) c h w -> b t c h w',b=B)
        if vid.ndim == 4:
            vid = vid[0]
        return vid

@clean_code.add_methods_from(inds_buffer)
class CES(nn.Module):

    def __init__(self, in_channels, num=6,
                 return_inds=False, attn_timer=False,
                 block_cfgs=None):
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

        # block_cfgs_l = config_to_list(search_cfg)
        kwargs = {"in_channels":in_channels,
                  "out_channels":in_channels}
        # print(block_cfgs[0])
        # print(block_cfgs[1])
        # print(block_cfgs[2])
        assert len(block_cfgs) == 3
        kwargs['search_cfg'] = block_cfgs[0]['search']
        for i in range(3):
            search_cfg_i = block_cfgs[i]['search']
            setattr(self,"c%d"%(i+1),merge_block(search_cfg_i,in_channels,in_channels))
        # self.c1 = merge_block(# **kwargs)
        # kwargs['search_cfg'] = block_cfgs[1]['search']
        # self.c2 = merge_block(**kwargs)
        # kwargs['search_cfg'] = block_cfgs[2]['search']
        # self.c3 = merge_block(**kwargs)
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

    def forward(self, vid, flows=None, inds=None, batchsize=1):
        self.clear_inds_buffer()

        state = [inds,None]
        out = self.c1(vid,flows,state,batchsize)
        inds0 = state[0]
        out = self.RBS1(out)
        out = self.c2(out,flows,state,batchsize)
        inds1 = state[0]
        out = self.RBS2(out)
        out = self.c3(out,flows,state,batchsize)
        inds2 = state[0]
        # if not(inds is None):
        #     inds0,inds1,inds2 = inds[0],inds[1],inds[2]
        #     out,inds0 = self.c1(vid,flows,inds0)
        #     out = self.RBS1(out)
        #     out,inds1 = self.c2(out,flows,inds1)
        #     out = self.RBS2(out)
        #     out,inds2 = self.c3(out,flows,inds2)
        # else:
        #     out,inds0 = self.c1(vid,flows,inds)
        #     out = self.RBS1(out)
        #     out,inds1 = self.c2(out,flows,inds0)
        #     out = self.RBS2(out)
        #     out,inds2 = self.c3(out,flows,inds1)
        inds = self.format_inds(inds0,inds1,inds2)
        self.update_ca_times()
        return out#,inds

