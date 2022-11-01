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
import colanet.configs as configs
import colanet.explore_configs as explore_configs


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

def load_results(cfg):

    # -- load cache --
    colanet_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(colanet_home / ".cache_io")
    cache_name = "explore_grid" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- set config --
    cfg = merge_with_base(cfg)
    cfg.bw = True
    # cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    # cfg.dname = "set8"
    # cfg.sigma = 10.
    # cfg.vid_name = "sunflower"
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    cfg.flow = "true"
    # cfg.isize = "128_128"
    cfg.adapt_mtype = "rand"
    cfg.ca_fwd = "dnls_k"
    cfg.use_train = "true"
    cfg.use_chop = "false"

    # -- config grid [1/3] --
    exps_a = explore_configs.search_space_cfg()
    # ws,wt,k,sb = [10,15,20,25,30],[0,1,2,3,5],[100],[256,1024,10*1024]
    ws,wt,k,sb = [20],[3],[100],[256,1024,10*1024]
    isize = ["128_128","256_256"]
    exp_lists = {"ws":ws,"wt":wt,"k":k, "sb":sb, "isize": isize}
    # exps_a = cache_io.mesh_pydicts(exp_lists) # get grid
    # cache_io.append_configs(exps_a,cfg)

    # -- config grid [2/3] --
    exp_lists['ws'] = [5]
    exp_lists['wt'] = [2,3,5]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg)

    # -- config grid [3/3] --
    exp_lists['ws'] = [20]
    exp_lists['wt'] = [0,3]
    exp_lists['k'] = [100]
    exp_lists['isize'] = ["96_96","118_118","140","164"]
    exp_lists['sb'] = [256,1024,3*1024,5*1024,10*1024]
    exps_c = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_c,cfg)

    # -- config grid [4/3] --
    exp_lists['ws'] = [20]
    exp_lists['wt'] = [3]
    exp_lists['k'] = [100]
    # exp_lists['sb'] = [220*220*3,180*180*3,140*140*3,100*100*3,60*60*3]
    # exp_lists['isize'] = ["220_220","180_180","140_140","100_100","60_60"]
    exp_lists['isize'] = ["512_512","384_384","256_256","128_128","64_64"]
    exp_lists['sb'] = [512*512*3,384*384*3,256*256*3,128*128*3,64*64*3]
    exps_d = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.nframes = 3
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cache_io.append_configs(exps_d,cfg)

    # -- combine --
    exps = exps_a# + exps_d# + exps_b + exps_c
    pp.pprint(exps[0])
    # for i in range(len(exps)):
    #     pp.pprint(exps[i])
    # exit(0)

    # -- read --
    root = Path("./.aaai23")
    if not root.exists():
        root.mkdir()
    pickle_store = str(root / "colanet_explore_grid.pkl")
    records = cache.load_flat_records(exps,save_agg=pickle_store,clear=False)

    # -- standardize col names --
    records = records.rename(columns={"sb":"batch_size"})

    return records
