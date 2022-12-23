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
import colanet.utils.gpu_mem as gpu_mem

# -- clean code --
from colanet.utils import clean_code
__methods__ = []
register_method = clean_code.register_method(__methods__)

@register_method
def forward_nl(self, vid, flows=None, inds_pred=None):

    # -- new dim --
    vid = vid[None,:]
    B = vid.shape[0]

    # -- init inds & search --
    self.clear_inds_buffer()
    self.update_search(inds_pred is None)

    # -- init timer --
    timer = ExpTimer(self.use_timer)
    timer.sync_start("attn")

    # -- batching params --
    nbatch,nbatches,ntotal = self.batching_info(vid.shape)

    # -- get images --
    timer.sync_start("extract")
    b1 = self.g(vid[0])[None,:]
    b2 = self.theta(vid[0])[None,:]
    b3 = self.phi(vid[0])[None,:]
    timer.sync_stop("extract")
    # print("a.")
    # print(b1[0,0,:3,:3,:3])
    # print(b2[0,0,:3,:3,:3])
    # print(b3[0,0,:3,:3,:3])
    # print("-"*30)

    # -- init & update --
    ifold = self.init_ifold(b1.shape,b1.device)
    if not(self.refine_inds):
        # print(vid.shape,flows.fflow.shape,flows.bflow.shape)
        self.search.update_flow(vid.shape,vid.device,flows)

    # -- batch across queries --
    # print(nbatch,nbatches,ntotal)
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
        # print(inds[0,0,:1])
        # print(dists)

        # -- subset to only aggregate --
        inds_agg = inds[...,:self.k_a,:].contiguous()
        dists_agg = dists[...,:self.k_a].contiguous()
        # print("inds_agg.shape: ",inds_agg.shape)
        # print("dists_agg.shape: ",dists_agg.shape)
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
    # print("vid.shape: ",vid.shape)
    # print("y.shape: ",y.shape)

    # -- final transform --
    y = self.W(y)
    # print("[c] y.shape: ",y.shape)
    # exit(0)
    y = vid + y

    # -- final mods --
    if self.add_SE:
        y_SE=self.SE(y)
        y=self.conv33(torch.cat((y_SE*y,y),dim=1))

    # -- get inds --
    inds = self.get_inds_buffer()
    # print("[final] inds.shape: ",inds.shape)
    self.clear_inds_buffer()

    # -- timer --
    timer.sync_stop("attn")
    # print(timer)
    if timer.use_timer:
        self.update_times(timer)
    # print("a.")
    # print(y[0,:3,:3,:3])
    # print("-"*20)

    return y,inds
