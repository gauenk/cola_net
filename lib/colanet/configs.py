"""

Default Configs for Training/Testing

"""

# -- easy dict --
import random
import numpy as np
import torch as th
from easydict import EasyDict as edict

def default_test_vid_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 0
    cfg.frame_start = 0
    cfg.frame_end = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/colanet/output/checkpoints/"
    # cfg.isize = "128_128"#None
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.mtype = "gray"
    cfg.bw = True
    cfg.seed = 123
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    return cfg


def default_train_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/colanet/output/checkpoints/"
    cfg.num_workers = 2
    cfg.device = "cuda:0"
    cfg.batch_size = 1
    cfg.batch_size_val = 1
    cfg.batch_size_te = 1
    cfg.saved_dir = "./output/saved_results/"
    cfg.device = "cuda:0"
    cfg.dname = "davis"
    cfg.mtype = "gray"
    cfg.bw = True
    cfg.nsamples_at_testing = 2
    cfg.nsamples_tr = 0
    cfg.nsamples_val = 2
    cfg.rand_order_val = False
    cfg.index_skip_val = 5
    cfg.nepochs = 10
    cfg.ensemble = "false"
    cfg.log_root = "./output/log"
    cfg.cropmode = "region_sobel"
    cfg.seed = 123
    return cfg

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
