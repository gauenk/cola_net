
# -- code api --
from . import original
from . import refactored
from . import batched
from . import configs
from . import lightning
from . import flow
from . import augmented

# -- publication api --
from . import aaai23
from . import cvpr23

# -- model api --
from .utils import optional

def load_model(cfg):
    mtype = optional(cfg,'model_type','augmented')
    print(mtype)
    if mtype == "augmented":
        return augmented.load_model(cfg)
    elif mtype == "refactored":
        nchnls = 1
        return refactored.load_model(cfg,cfg.mtype,2,nchnls)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
