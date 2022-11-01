"""

I am an API to write the paper for AAAI

"""


# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- caching results --
import cache_io

# -- network --
import colanet

def append_detailed_cfg(cfg):
    # -- append default --
    cfg_l = [cfg]
    def_cfg = colanet.configs.default_test_vid_cfg()
    def_cfg.nframes = cfg.nframes
    def_cfg.frame_start = cfg.frame_start
    def_cfg.frame_end = cfg.frame_end
    def_cfg.isize = cfg.isize
    cache_io.append_configs(cfg_l,def_cfg) # merge the two
    cfg = cfg_l[0]

    # -- get config --
    cfg.adapt_mtype = "rand"
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    cfg.ws = 20
    cfg.wt = 3
    cfg.bw = True
    cfg.k = 100

    return cfg

def load_proposed(cfg,use_train="true",flow="true"):
    use_chop = "false"
    ca_fwd = "dnls_k"
    sb = 48*1024
    return load_results(ca_fwd,use_train,use_chop,flow,sb,cfg)

def load_original(cfg,use_chop="true"):
    flow = "false"
    use_train = "false"
    ca_fwd = "default"
    sb = 1
    return load_results(ca_fwd,use_train,use_chop,flow,sb,cfg)

def load_results(ca_fwd,use_train,use_chop,flow,sb,cfg):

    # -- get cache --
    colanet_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(colanet_home / ".cache_io")
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- update with default --
    cfg = append_detailed_cfg(cfg)
    cfg.flow = flow
    cfg.ca_fwd = ca_fwd
    cfg.use_train = use_train
    cfg.use_chop = use_chop
    cfg.sb = sb
    cfg.frame_end = -1

    # -- load results --
    cfg_l = [cfg]
    pp.pprint(cfg_l[0])
    records = cache.load_flat_records(cfg_l)
    records['home_path'] = colanet_home
    return records
