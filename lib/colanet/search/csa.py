import torch as th
from torch.nn.functional import fold,unfold
from einops import rearrange,repeat
from ..utils.proc_utils import expand2square


def iunfold(vid,ps,stride):
    patches = unfold(vid,(ps,ps),stride=stride).transpose(-2, -1)
    return patches

def init_from_cfg(cfg):
    return CSASearch(cfg.ps,cfg.nheads,cfg.stride0,cfg.stride1)

class CSASearch():

    def __init__(self,ps=7, nheads=1, stride0=1, stride1=1):
        self.ps = ps
        self.nheads = nheads
        self.stride0 = stride0
        self.stride1 = stride1

    def __call__(self,vid,*args):
        nheads = self.nheads
        B,T,C,H,W = vid.shape
        vid = vid
        vid = rearrange(vid,'b t (H c) h w -> (b t H) c h w',H=nheads)
        patches = iunfold(vid,self.ps,self.stride0)
        q = patches
        k = iunfold(vid,self.ps,self.stride1)
        attn = th.matmul(q,k.transpose(-2, -1))
        attn = rearrange(attn,'(b t H) d0 d1 -> b t H d0 d1',b=B,H=nheads)
        inds = th.zeros((1))
        return attn,inds

    def flops(self,B,C,H,W):

        # -- init --
        num = B * self.nheads
        ps = self.ps
        dim = ps*ps*(C//self.nheads)
        nH0 = (H-1)//self.stride0+1
        nW0 = (W-1)//self.stride0+1
        nH1 = (H-1)//self.stride1+1
        nW1 = (W-1)//self.stride1+1
        N0 = nH0*nW0
        N1 = nH1*nW1
        nflops = num * N0 * N1 * (dim + dim)
        return nflops

    def radius(self,H,W):
        return max(H,W)

