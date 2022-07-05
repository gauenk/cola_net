
import numpy as np
from easydict import EasyDict as edict

def select_sigma(sigma):
    sigmas = np.array([10, 15, 25, 30, 50, 70])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def get_options():
    return option.args

def default_options(sigma=0.):
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
    args.sigma = sigma
    return args

def crop_offset(in_image, row_offs, col_offs):
    if len(row_offs) == 1: row_offs += row_offs
    if len(col_offs) == 1: col_offs += col_offs

    if row_offs[1] > 0 and col_offs[1] > 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], col_offs[0]:-col_offs[-1]]
        # t,c,h,w = in_image.shape
        # hr,wr = h-2*row_offs[0],w-2*col_offs[0]
        # out_image = tvf.center_crop(in_image,(hr,wr))
    elif row_offs[1] > 0 and col_offs[1] == 0:
        raise NotImplemented("")
        # out_image = in_image[..., row_offs[0]:-row_offs[1], :]
    elif 0 == row_offs[1] and col_offs[1] > 0:
        raise NotImplemented("")
        # out_image = in_image[..., :, col_offs[0]:-col_offs[1]]
    else:
        out_image = in_image
    return out_image
