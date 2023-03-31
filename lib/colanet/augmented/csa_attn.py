
# -- mics --
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
from torch.nn.functional import fold as th_fold
from .tiling import *
from dev_basics.utils.timer import ExpTimerList,ExpTimer
from functools import partial

# -- clean code --
from colanet.utils import clean_code
__methods__ = []
register_method = clean_code.register_method(__methods__)

@register_method
def forward_csa(self, vid, flows=None, inds_pred=None):

    # -- new dim --
    vid = vid[None,:]
    B = vid.shape[0]

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

    # -- init & update --
    ps = self.ps
    T,C,H,W = b2[0].shape
    # ifold = self.init_ifold(b1.shape,b1.device)
    pad = ps//2
    # unfold0 = nn.Unfold(self.ps,stride=self.stride0,padding=pad)
    # unfold1 = nn.Unfold(self.ps,stride=self.stride1,padding=pad)
    unfold0 = partial(vid2patches,self.ps,self.stride0)
    unfold1 = partial(vid2patches,self.ps,self.stride1)

    # -- batch across queries --
    assert nbatches == 1
    for index in range(nbatches):

        # -- batch info --
        qindex = min(nbatch * index,ntotal)
        nbatch_i =  min(nbatch, ntotal - qindex)

        # -- search --
        timer.sync_start("search")
        p0 = rearrange(unfold0(b1[0])[0],'t f n -> t n f')
        p1,pad = unfold1(b3[0])
        p1 = rearrange(p1,'t f n -> t f n')
        dists = p0 @ p1
        timer.sync_stop("search")
        # print("inds.shape: ",inds.shape)
        # print(dists)

        # -- normalize --
        timer.sync_start("normalize")
        weights = F.softmax(dists*self.softmax_scale, dim=1)
        assert_nonan(weights)
        timer.sync_stop("normalize")

        # -- attn mask --
        timer.sync_start("agg")
        p3 = unfold1(b2[0])[0]
        p3 = rearrange(p3,'t f n -> t n f')
        _T = weights.shape[0]
        zi = []
        for ti in range(_T):
            zi_i = th.mm(weights[ti],p3[ti])
            zi.append(zi_i)
        zi = th.stack(zi,0)
        timer.sync_stop("agg")

        # -- ifold --
        timer.sync_start("fold")
        # zi = rearrange(zi,'n (ph pw c) b H q c h w -> b q H 1 c h w')
        zi = rearrange(zi,'t n f -> t f n')
        ones = th.ones_like(zi)
        yvid = th_fold(zi,(H,W),(ps,ps),padding=pad[0],stride=self.stride0)
        zvid = th_fold(ones,(H,W),(ps,ps),padding=pad[0],stride=self.stride0)
        y = yvid / zvid
        # print(y)
        assert_nonan(y)
        # ifold(zi,qindex)
        timer.sync_stop("fold")
        # print(timer)

    # -- get post-attn vid --
    # y,Z = ifold.vid,ifold.zvid
    # y = y / Z

    # -- remove batching --
    vid = vid[0]

    # -- final transform --
    y = self.W(y)
    y = vid + y
    assert_nonan(y)

    # -- final mods --
    if self.add_SE:
        y_SE=self.SE(y)
        y=self.conv33(torch.cat((y_SE*y,y),dim=1))

    # -- timer --
    timer.sync_stop("attn")
    if timer.use_timer:
        self.update_times(timer)
    inds = th.empty(0)
    return y#,inds

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings

def vid2patches(ksizes, strides, images):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    ksizes = [ksizes,ksizes]
    strides = [strides,strides]
    images, paddings = same_padding(images, ksizes, strides, [1,1])
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings

