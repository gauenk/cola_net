
import torch as th
import pickle
from easydict import EasyDict as edict
from einops import rearrange


def fwd_4dim(fxn,vid):
    if vid.ndim == 4:
        return fxn(vid)
    B,T = vid.shape[:2]
    vid = rearrange(vid,'b t c h w -> (b t) c h w')
    out = fxn(vid)
    out = rearrange(out,'(b t) c h w -> b t c h w',b=B)
    return out

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def optional_delete(pydict,key):
    if pydict is None: return
    elif key in pydict: del pydict[key]
    else: return

def assert_nonan(tensor):
    assert th.any(th.isnan(tensor)).item() is False

def rslice(vid,coords):
    if coords is None: return vid
    if len(coords) == 0: return vid
    if th.is_tensor(coords):
        coords = coords.type(th.int)
        coords = list(coords.cpu().numpy())
    fs,fe,t,l,b,r = coords
    return vid[fs:fe,:,t:b,l:r]

def slice_flows(flows,t_start,t_end):
    if flows is None: return flows
    flows_t = edict()
    flows_t.fflow = flows.fflow[t_start:t_end]
    flows_t.bflow = flows.bflow[t_start:t_end]
    return flows_t

def write_pickle(fn,obj):
    with open(str(fn),"wb") as f:
        pickle.dump(obj,f)

def read_pickle(fn):
    with open(str(fn),"rb") as f:
        obj = pickle.load(f)
    return obj
