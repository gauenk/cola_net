
import torch as th
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict

from .model import Model

def select_sigma(sigma):
    sigmas = np.array([10, 15, 25, 30, 50, 70])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def load_model(data_sigma,version=2):

    # -- params --
    model_sigma = select_sigma(data_sigma)
    args = default_options()
    args.ensemble = False
    checkpoint = edict()
    checkpoint.dir = "."

    # -- weights --
    weights = Path("/home/gauenk/Documents/packages/colanet/weights/checkpoints/")
    weights /= ("DN_Gray/res_cola_v%d_6_3_%d_l4/" % (version,model_sigma))
    weights /= "model/model_best.pt"
    weights = Path(weights)

    # -- model --
    model = Model(args,checkpoint)
    model.model.load_state_dict(th.load(weights,map_location='cuda'))
    model.eval()
    return model

def get_options():
    return option.args

def default_options():
    args = edict()
    args.scale = [1]
    args.self_ensemble = False
    args.chop = True
    args.precision = "single"
    args.cpu = False
    args.n_GPUs = 1
    args.pre_train = "."
    args.save_models = False
    args.model = "COLA"
    args.mode = "E"
    args.print_model = False
    args.resume = 0
    args.seed = 1
    args.n_resblocks = 16
    args.n_feats = 64
    args.n_colors = 1
    args.res_scale = 1
    args.rgb_range = 1.
    args.stages = 6
    args.blocks = 3
    args.act = "relu"
    return args
