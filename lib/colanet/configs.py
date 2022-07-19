"""

Default Configs for Training/Testing

"""

# -- easy dict --
from easydict import EasyDict as edict

def default_test_vid_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 0
    cfg.frame_start = 0
    cfg.frame_end = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/colanet/output/checkpoints/"
    cfg.isize = "128_128"#None
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.mtype = "gray"
    cfg.bw = True
    return cfg


def default_train_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/colanet/output/checkpoints/"
    cfg.num_workers = 2
    cfg.device = "cuda:0"
    cfg.batch_size = 1
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
    return cfg


