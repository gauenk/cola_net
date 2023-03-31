
import dnls
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as dcopy
from torch.autograd import gradcheck
from einops import rearrange,repeat
from colanet.utils.misc import assert_nonan
import colanet
from colanet.utils import optional
from torch.nn.functional import unfold as th_unfold
from .tiling import *
from dev_basics.utils.timer import ExpTimerList,ExpTimer

# -- modules --
from . import inds_buffer
from . import attn_mods
from . import csa_attn
from . import nl_attn
from dev_basics.utils import clean_code


"""
CA network
"""
@clean_code.add_methods_from(nl_attn)
@clean_code.add_methods_from(csa_attn)
@clean_code.add_methods_from(attn_mods)
@clean_code.add_methods_from(inds_buffer)
class ContextualAttention_Enhance(nn.Module):

    def __init__(self, search_cfg, in_channels=64, inter_channels=16,
                 use_multiple_size=False,add_SE=False,
                 softmax_scale=10,shape=64,p_len=64,
                 refine_inds=False,return_inds=False,attn_timer=False):
        super(ContextualAttention_Enhance, self).__init__()

        self.shape=shape
        self.p_len=p_len
        # self.stride0 = stride0
        # self.stride1 = stride1
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.add_SE=add_SE
        self.use_inds_buffer = return_inds
        self.search_name = search_cfg.search_name
        self.batchsize = search_cfg.batchsize
        self.ps = search_cfg.ps
        self.inds_buffer = []
        self.search_cfg = search_cfg

        # -- se layer --
        self.conv33 = None
        self.SE = None
        if self.add_SE:
            self.SE=SE_net(in_channels=in_channels)
            self.conv33=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,
                                  kernel_size=1,stride=1,padding=0)

        # -- xforms; q,k,v --
        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels,
                           out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        # -- assign --
        # self.attn_mode = attn_mode
        # self.search_name = search_name
        # self.k_s = k_s
        # self.k_a = k_a
        # self.ps = ps
        # self.pt = pt
        # self.ws = ws
        # self.wr = wr
        # self.kr = kr
        # self.wt = wt
        self.stride0 = search_cfg.stride0
        self.stride1 = search_cfg.stride1
        self.use_state_update = search_cfg.use_state_update
        # self.dilation = dilation
        # self.rbwd = rbwd
        # self.nbwd = nbwd
        # self.exact = exact
        # self.reflect_bounds = reflect_bounds
        # self.refine_inds = refine_inds
        # self.search_kwargs = {"k":k_s,"ps":ps,
        #                       "pt":pt,"ws":ws,"wt":wt,"wr":wr,"kr":kr,
        #                       "stride0":stride0,"stride1":stride1,
        #                       "dilation":dilation,"rbwd":rbwd,"nbwd":nbwd,
        #                       "exact":exact,"reflect_bounds":reflect_bounds,
        #                       "refine_inds":refine_inds}
        self.search = None
        if search_cfg.search_name != "csa":
            self.search = dnls.search.init(search_cfg)

        # self.search = self.init_search(attn_mode=attn_mode,k=k_s,ps=ps,pt=pt,
        #                                ws=ws,ws_r=ws_r,wt=wt,
        #                                stride0=stride0,stride1=stride1,
        #                                dilation=dilation,rbwd=rbwd,nbwd=nbwd,
        #                                exact=exact,reflect_bounds=reflect_bounds,
        #                                refine_inds=refine_inds)
        # self.wpsum = self.init_wpsum(ps=ps,pt=pt,dilation=dilation,
        #                              reflect_bounds=reflect_bounds,exact=exact)
        agg_fields = ["ps","pt","dilation","reflect_bounds","exact"]
        agg_cfg = {k:search_cfg[k] for k in agg_fields}
        self.wpsum = self.init_wpsum(**agg_cfg)

        # -- timers --
        # self.times = AggTimer()
        # self.timer = ExpTimer(attn_timer)
        self.use_timer = attn_timer
        self.times = ExpTimerList(self.use_timer)

    def forward(self, vid, flows=None, inds_pred=None, batchsize=1):
        if self.search_name == "csa":
            vid = self.forward_csa(vid,flows,inds_pred) # csa_attn.py
        else:
            vid = rearrange(vid,'(b t) c h w -> b t c h w',b=batchsize)
            vid = self.forward_nl(vid,flows,inds_pred) # nl_attn.py
            vid = rearrange(vid,'b t c h w -> (b t) c h w')
        return vid

    def GSmap(self,a,b):
        return torch.matmul(a,b)


class SE_net(nn.Module):
    def __init__(self,in_channels,reduction=16):
        super(SE_net,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//reduction,kernel_size=1,stride=1,padding=0)
        self.fc2=nn.Conv2d(in_channels=in_channels//reduction,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        o1=self.pool(x)
        o1=F.relu(self.fc1(o1))
        o1=self.fc2(o1)
        return o1


