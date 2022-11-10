import numpy as np

from .dn_gray.io import load_model as load_model_gray
from .dn_real.io import load_model as load_model_real
from ..utils.misc import optional,optional_delete

def load_model(cfg,name,version=2,nchnls=1):
    # name = optional(kwargs,'name','real')
    # optional_delete(kwargs,'name')
    if name == "real":
        model = load_model_real(version)
        return model
    elif name == "gray":
        model = load_model_gray(cfg,version,nchnls)
        return model
    else:
        raise ValueError(f"Uknown model name [{name}]")
