
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

    def __init__(self, in_channels=64, inter_channels=16,
                 use_multiple_size=False,add_SE=False,
                 softmax_scale=10,shape=64,p_len=64,
                 attn_mode="dnls_k",k_s=100,k_a=100,ps=7,pt=0,
                 ws=21,ws_r=3,wt=0,stride0=4,stride1=1,dilation=1,bs=-1,
                 rbwd=True,nbwd=1,exact=False,reflect_bounds=False,
                 refine_inds=False,return_inds=False,attn_timer=False):
        super(ContextualAttention_Enhance, self).__init__()

        self.shape=shape
        self.p_len=p_len
        self.stride0 = stride0
        self.stride1 = stride1
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.add_SE=add_SE
        self.bs = bs
        self.use_inds_buffer = return_inds
        self.inds_buffer = []

        self.conv33=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,
                              kernel_size=1,stride=1,padding=0)
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
        self.attn_mode = attn_mode
        self.k_s = k_s
        self.k_a = k_a
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.ws_r = ws_r
        self.wt = wt
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.rbwd = rbwd
        self.nbwd = nbwd
        self.exact = exact
        self.reflect_bounds = reflect_bounds
        self.refine_inds = refine_inds
        self.search_kwargs = {"attn_mode":attn_mode,"k":k_s,"ps":ps,
                              "pt":pt,"ws":ws,"ws_r":ws_r,"wt":wt,
                              "stride0":stride0,"stride1":stride1,
                              "dilation":dilation,"rbwd":rbwd,"nbwd":nbwd,
                              "exact":exact,"reflect_bounds":reflect_bounds,
                              "refine_inds":refine_inds}
        self.search = self.init_search(attn_mode=attn_mode,k=k_s,ps=ps,pt=pt,
                                       ws=ws,ws_r=ws_r,wt=wt,
                                       stride0=stride0,stride1=stride1,
                                       dilation=dilation,rbwd=rbwd,nbwd=nbwd,
                                       exact=exact,reflect_bounds=reflect_bounds,
                                       refine_inds=refine_inds)
        self.wpsum = self.init_wpsum(ps=ps,pt=pt,dilation=dilation,
                                     reflect_bounds=reflect_bounds,exact=exact)

        # -- timers --
        # self.times = AggTimer()
        # self.timer = ExpTimer(attn_timer)
        self.use_timer = attn_timer
        self.times = ExpTimerList(self.use_timer)

    def forward(self, vid, flows=None, inds_pred=None):
        if "dnls" in self.attn_mode:
            return self.forward_nl(vid,flows,inds_pred) # nl_attn.py
        elif self.attn_mode == "csa":
            return self.forward_csa(vid,flows,inds_pred) # csa_attn.py
        else:
            raise ValueError(f"Uknown attention mode [{self.attn_mode}]")

    def GSmap(self,a,b):
        return torch.matmul(a,b)


