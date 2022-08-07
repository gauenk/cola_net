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

def load_intro(cfg):

    # -- get cache --
    lidia_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(lidia_home / ".cache_io")
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # dnames,sigmas = ["set8"],[50.]
    # vid_names = ["sunflower"]
    # # vid_names = ["sunflower","hypersmooth","tractor"]
    # # vid_names = ["snowboard","sunflower","tractor","motorbike",
    # #              "hypersmooth","park_joy","rafting","touchdown"]
    # internal_adapt_nsteps = [300]
    # internal_adapt_nepochs = [0]
    # ws,wt,k,sb = [20],[3],[100],[1024*32]
    # flow,isizes,adapt_mtypes = ["true"],["none"],["rand"]
    # ca_fwd_list,use_train = ["dnls_k"],["true"]
    # exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
    #              "internal_adapt_nsteps":internal_adapt_nsteps,
    #              "internal_adapt_nepochs":internal_adapt_nepochs,
    #              "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
    #              "isize":isizes,"use_train":use_train,"ca_fwd":ca_fwd_list,
    #              "ws":ws,"wt":wt,"k":k, "sb":sb, "use_chop":["false"]}

    # -- update with default --
    cfg_l = [cfg]
    def_cfg = colanet.configs.default_test_vid_cfg()
    def_cfg.isize = "256_256"
    cache_io.append_configs(cfg_l,def_cfg) # merge the two
    cfg = cfg_l[0]

    # -- get config --
    cfg.nframes = 2
    cfg.frame_start = 10
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.adapt_mtype = "rand"
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    cfg.ws = 20
    cfg.wt = 3
    cfg.sb = 32*1024
    cfg.bw = True
    cfg.k = 100
    cfg.ca_fwd = "dnls_k"
    cfg.use_train = "true"
    cfg.flow = "true"
    cfg.use_chop = "false"
    cfg_l = [cfg]

    # -- load results --
    # pp.pprint(cfg_l[0])
    records = cache.load_flat_records(cfg_l)
    return records
