"""

I am an API to write the paper for CVPR

"""

# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)
import copy
dcopy = copy.deepcopy

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

"""
{   'attn_mode': 'stnls_k',
    'bs': 49152,
    'bw': True,
    'ca_fwd': 'stnls_k',
    'checkpoint_dir': '/home/gauenk/Documents/packages/colanet/output/checkpoints/',
    'device': 'cuda:0',
    'dname': 'set8',
    'flow': 'true',
    'frame_end': 2,
    'frame_start': 0,
    'internal_adapt_nepochs': 0,
    'internal_adapt_nsteps': 300,
    'isize': '128_128',
    'k': 100,
    'mtype': 'gray',
    'nframes': 3,
    'num_workers': 1,
    'return_inds': True,
    'saved_dir': './output/saved_results/',
    'seed': 123,
    'sigma': 50,
    'spatial_crop_overlap': 0.0,
    'spatial_crop_size': 'none',
    'temporal_crop_overlap': 0.0,
    'temporal_crop_size': 3,
    'use_chop': 'false',
    'use_train': 'true',
    'vid_name': 'sunflower',
    'ws': 29,
    'wt': 3}
"""

def detailed_cfg(cfg):

    # -- append default --
    # cfg_l = [cfg]
    # def_cfg = configs.default_test_vid_cfg()

    # -- data config --
    cfg.isize = "256_256"
    cfg.bw = True
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.return_inds = True
    cfg.ca_fwd = 'stnls_k'

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 3#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.attn_mode = "stnls_k"
    cfg.use_chop = "false"

    # dnames,sigmas = ["set8"],[50]#,30.]
    # # vid_names = ["tractor"]
    # vid_names = ["sunflower"]
    # # vid_names = ["sunflower","hypersmooth","tractor"]
    # # vid_names = ["snowboard","sunflower","tractor","motorbike",
    # #              "hypersmooth","park_joy","rafting","touchdown"]
    # ws,wt,k,bs = [29],[3],[100],[48*1024]
    # flow,isizes = ["true"],["none"]
    # ca_fwd_list = ["stnls_k"]
    # use_train = ["true","false"]

    # -- append --
    # cache_io.append_configs(cfg_l,cfg) # merge the two
    # cfg = cfg_l[0]

    # -- get config --
    cfg.ws = 29
    cfg.wt = 3
    cfg.bw = True
    cfg.k = 100
    cfg.bs = 48*1024
    # return cfg


def merge_with_base(cfg):
    # -- [first] merge with base --
    cfg_og = dcopy(cfg)
    cfg_l = [cfg]
    cfg_base = colanet.configs.default_test_vid_cfg()
    cache_io.append_configs(cfg_l,cfg_base)
    cfg = cfg_l[0]

    # -- overwrite with input values --
    for key in cfg_og:
        cfg[key] = cfg_og[key]

    # -- remove extra keys --
    del cfg['isize']
    return cfg

# def load_proposed(cfg,use_train="true",flow="true"):
#     use_chop = "false"
#     ca_fwd = "stnls_k"
#     sb = 256
#     return load_results(ca_fwd,use_train,use_chop,flow,sb,cfg)

# def load_original(cfg,use_chop="false"):
#     flow = "false"
#     use_train = "false"
#     ca_fwd = "default"
#     sb = 1
#     return load_results(ca_fwd,use_train,use_chop,flow,sb,cfg)

def load_results(cfg):

    # -- get cache --
    home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(home / ".cache_io")
    cache_name = "show_refine" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- update with default --
    cfg = merge_with_base(cfg)
    detailed_cfg(cfg)
    exp_lists = {}
    exp_lists['wt'] = [3]
    exp_lists['ws'] = [29]
    exp_lists['use_train'] = ["true","false"]
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps,cfg)
    pp.pprint(exps[0])

    # -- read --
    root = Path("./.cvpr23")
    if not root.exists():
        root.mkdir()
    pickle_store = str(root / "colanet_show_refine.pkl")
    records = cache.load_flat_records(exps,save_agg=pickle_store,clear=True)

    # -- standardize col names --
    records = records.rename(columns={"bs":"batch_size"})
    return records
