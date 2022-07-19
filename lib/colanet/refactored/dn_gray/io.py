
import copy
import torch as th
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict

from .model import Model
from .misc import select_sigma,default_options

def load_model(data_sigma,version=1,chnls=1):

    # -- params --
    model_sigma = select_sigma(data_sigma)
    args = default_options()
    args.ensemble = False
    checkpoint = edict()
    checkpoint.dir = "."
    args.n_colors = chnls

    # -- weights --
    weights = Path("/home/gauenk/Documents/packages/colanet/weights/checkpoints/")
    weights /= ("DN_Gray/res_cola_v%d_6_3_%d_l4/" % (version,model_sigma))
    weights /= "model/model_best.pt"
    weights = Path(weights)

    # -- model --
    model = Model(args,checkpoint)
    model_state = th.load(weights,map_location='cuda')
    # modded_dict(model_state)
    model.model.load_state_dict(model_state)
    model.eval()
    return model

def modded_dict(mdict):
    names = sorted(list(mdict.keys()))
    for name in names:
        name_og = copy.copy(name)
        for k in range(20,-1,-1):
            sname = name.split(".")
            rename = False
            try:
                int(sname[1])
                rename = True
            except:
                rename = False
            if rename:
                new_s = sname[0] + sname[1]
                keep = sname[2:]
                new = [new_s] + keep
                name = ".".join(new)
        value = mdict[name_og]
        del mdict[name_og]
        mdict[name] = value

