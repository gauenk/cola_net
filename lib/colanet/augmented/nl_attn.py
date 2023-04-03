import stnls
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as dcopy
from torch.autograd import gradcheck
from einops import rearrange,repeat
from colanet.utils.misc import assert_nonan
import colanet
from colanet.utils import optional,fwd_4dim
from torch.nn.functional import unfold as th_unfold
from .tiling import *
from dev_basics.utils.timer import ExpTimerList,ExpTimer
import colanet.utils.gpu_mem as gpu_mem

def print_nan_info(vid,y,Z,dists,inds,state,search_cfg):
    print(search_cfg)
    print("y.shape: ",y.shape)
    print("vid.shape: ",vid.shape)
    print("All Z > 0? ",th.all(Z>0))
    print("Any inds < 0? ",th.any(inds < 0))

# -- clean code --
from colanet.utils import clean_code
__methods__ = []
register_method = clean_code.register_method(__methods__)

@register_method
def run_search(self,q_vid,k_vid,flows,state):
    self.timer.sync_start("search")
    if self.search_name == "refine":
        inds_p = self.inds_rs1(state[0])
        dists,inds = self.search(q_vid,k_vid,inds_p)
    elif self.search_name == "rand_inds":
        dists,inds = self.search(q_vid,k_vid)
    else:
        dists,inds = self.search(q_vid,k_vid,flows.fflow,flows.bflow)
    self.update_state(state,dists,inds,q_vid.shape)
    self.timer.sync_stop("search")
    return dists,inds

@register_method
def update_state(self,state,dists,inds,vshape):
    if not(self.use_state_update): return
    T,C,H,W = vshape[-4:]
    nH = (H-1)//self.stride0+1
    nW = (W-1)//self.stride0+1
    state[0] = state[1]
    state[1] = self.inds_rs0(inds.detach(),nH,nW)

@register_method
def inds_rs0(self,inds,nH,nW):
    if not(inds.ndim == 5): return inds
    rshape = 'b h (T nH nW) k tr -> T nH nW b h k tr'
    inds = rearrange(inds,rshape,nH=nH,nW=nW)
    return inds

@register_method
def inds_rs1(self,inds):
    if not(inds.ndim == 7): return inds
    rshape = 'T nH nW b h k tr -> b h (T nH nW) k tr'
    inds = rearrange(inds,rshape)
    return inds

@register_method
def forward_nl(self, vid, flows=None, state=None):

    # -- new dim --
    # vid = vid[None,:]
    B = vid.shape[0]

    # -- init inds & search --
    # self.clear_inds_buffer()
    # self.update_search(inds_pred is None)

    # -- init timer --
    self.timer = ExpTimer(self.use_timer)
    self.timer.sync_start("attn")

    # -- batching params --
    # nbatch,nbatches,ntotal = self.batching_info(vid.shape)

    # -- get images --
    self.timer.sync_start("extract")
    b1 = fwd_4dim(self.g,vid)
    b2 = fwd_4dim(self.theta,vid)
    b3 = fwd_4dim(self.phi,vid)
    self.timer.sync_stop("extract")
    # print("a.")
    # print(b1[0,0,:3,:3,:3])
    # print(b2[0,0,:3,:3,:3])
    # print(b3[0,0,:3,:3,:3])
    # print("-"*30)

    # -- init & update --
    ifold = self.init_ifold(b1.shape,b1.device)
    # if not(self.refine_inds):
    #     # print(vid.shape,flows.fflow.shape,flows.bflow.shape)
    #     self.search.update_flow(vid.shape,vid.device,flows)

    # -- run search --
    self.timer.sync_start("search")
    # dists,inds = self.search.wrap_fwd(b1,qindex,nbatch_i,b3,inds_pred)
    # print(b1.shape,b3.shape)
    dists,inds = self.run_search(b1,b3,flows,state)
    self.timer.sync_stop("search")

    # -- subset to only aggregate --
    # if self.k_a > 0:
    #     inds_agg = inds[...,:self.k_a,:].contiguous()
    #     dists_agg = dists[...,:self.k_a].contiguous()
    inds_agg = inds.contiguous()#[...,:self.k_a,:].contiguous()
    dists_agg = dists.contiguous()#[...,:self.k_a].contiguous()


    # -- attn mask --
    self.timer.sync_start("agg")
    # print("softmax_scale: ",self.softmax_scale)
    # print(dists_agg.shape)
    yi = F.softmax(dists_agg*self.softmax_scale,-1)
    assert_nonan(yi)
    zi = self.wpsum(b2,yi,inds_agg)
    self.timer.sync_stop("agg")

    # -- ifold --
    self.timer.sync_start("fold")
    zi = rearrange(zi,'b H q c h w -> b q H 1 c h w')
    ifold(zi,0)#qindex)
    self.timer.sync_stop("fold")
    # print(timer)

    # -- update --
    # self.update_inds_buffer(inds)

    # # -- batch across queries --
    # # print(nbatch,nbatches,ntotal)
    # for index in range(nbatches):

    #     # -- batch info --
    #     qindex = min(nbatch * index,ntotal)
    #     nbatch_i =  min(nbatch, ntotal - qindex)
    #     # print(qindex)

    #     # -- search --
    #     # print("b1.shape,b3.shape: ",b1.shape,b3.shape)
    #     # print(self.search.ws,self.search.wt,self.search.ps,
    #     #       self.search.k,self.search.chnls,
    #     #       self.search.stride0,self.search.stride1)
    #     # print(self.search)
    #     # print("inds.shape: ",inds.shape)
    #     # print(inds[0,0,:1])
    #     # print(dists)


    # -- get post-attn vid --
    y,Z = ifold.vid,ifold.zvid
    y = y / Z
    if th.any(th.isnan(y)):
        print_nan_info(vid,y,Z,dists,inds,state,self.search_cfg)
        print("Nan found.")
        exit(0)
    # assert_nonan(y)

    # -- remove batching --
    vid = rearrange(vid,'b t c h w -> (b t) c h w')
    y = rearrange(y,'b t c h w -> (b t) c h w')

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
    self.timer.sync_stop("attn")
    # print(timer)
    if self.timer.use_timer:
        self.update_times(self.timer)
    # print("a.")
    # print(y[0,:3,:3,:3])
    # print("-"*20)

    y = rearrange(y,'(b t) c h w -> b t c h w',b=B)
    return y
