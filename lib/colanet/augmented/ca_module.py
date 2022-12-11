
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
from colanet.utils import ExpTimer,AggTimer

# -- modules --
from . import inds_buffer
from . import attn_mods
from colanet.utils import clean_code


"""
CA network
"""
@clean_code.add_methods_from(attn_mods)
@clean_code.add_methods_from(inds_buffer)
class ContextualAttention_Enhance(nn.Module):

    def __init__(self, in_channels=64, inter_channels=16,
                 use_multiple_size=False,add_SE=False,
                 softmax_scale=10,shape=64,p_len=64,
                 attn_mode="dnls_k",k_s=100,k_a=100,ps=7,pt=0,
                 ws=21,ws_r=3,wt=0,stride0=4,stride1=1,dilation=1,bs=-1,
                 rbwd=True,nbwd=1,exact=False,reflect_bounds=False,
                 refine_inds=False,return_inds=False):
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
        self.times = AggTimer()

    def forward(self, vid, flows=None, inds_pred=None):


        # -- new dim --
        vid = vid[None,:]
        B = vid.shape[0]

        # -- init inds & search --
        self.clear_inds_buffer()
        self.update_search(inds_pred is None)

        # -- init timer --
        use_timer = False
        timer = ExpTimer(use_timer)

        # -- batching params --
        nbatch,nbatches,ntotal = self.batching_info(vid.shape)

        # -- get images --
        timer.sync_start("extract")
        b1 = self.g(vid[0])[None,:]
        b2 = self.theta(vid[0])[None,:]
        b3 = self.phi(vid[0])[None,:]
        timer.sync_stop("extract")

        # -- init & update --
        ifold = self.init_ifold(b1.shape,b1.device)
        if not(self.refine_inds):
            # print(vid.shape,flows.fflow.shape,flows.bflow.shape)
            self.search.update_flow(vid.shape,vid.device,flows)

        # -- batch across queries --
        for index in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * index,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- search --
            # print("b1.shape,b3.shape: ",b1.shape,b3.shape)
            # print(self.search.ws,self.search.wt,self.search.ps,
            #       self.search.k,self.search.chnls,
            #       self.search.stride0,self.search.stride1)
            # print(self.search)
            timer.sync_start("search")
            dists,inds = self.search.wrap_fwd(b1,qindex,nbatch_i,b3,inds_pred)
            timer.sync_stop("search")
            # print("inds.shape: ",inds.shape)
            # print(dists)

            # -- subset to only aggregate --
            inds_agg = inds[...,:self.k_a,:].contiguous()
            dists_agg = dists[...,:self.k_a].contiguous()
            # print("inds_agg.shape: ",inds_agg.shape)
            # print(dists_agg)

            # -- attn mask --
            timer.sync_start("agg")
            # print("softmax_scale: ",self.softmax_scale)
            yi = F.softmax(dists_agg*self.softmax_scale,2)
            assert_nonan(yi)
            zi = self.wpsum(b2[:,None],yi[:,None],inds_agg[:,None])
            timer.sync_stop("agg")

            # -- ifold --
            timer.sync_start("fold")
            zi = rearrange(zi,'b H q c h w -> b q H 1 c h w')
            ifold(zi,qindex)
            timer.sync_stop("fold")
            # print(timer)

            # -- update --
            self.update_inds_buffer(inds)

        # -- get post-attn vid --
        y,Z = ifold.vid,ifold.zvid
        y = y / Z
        assert_nonan(y)

        # -- remove batching --
        vid,y = vid[0],y[0]

        # -- final transform --
        y = self.W(y)
        y = vid + y

        # -- final mods --
        if self.add_SE:
            y_SE=self.SE(y)
            y=self.conv33(torch.cat((y_SE*y,y),dim=1))

        # -- get inds --
        inds = self.get_inds_buffer()
        # print("[final] inds.shape: ",inds.shape)
        self.clear_inds_buffer()

        # -- viz --
        if timer.use_timer:
            # print(timer)
            self.update_timer(timer)

        return y,inds

    def GSmap(self,a,b):
        return torch.matmul(a,b)


