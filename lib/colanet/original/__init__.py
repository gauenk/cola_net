import numpy as np

from .dn_gray.io import load_model as load_model_gray
from .dn_real.io import load_model as load_model_real
from ..utils.misc import optional,optional_delete

def load_model(name,data_sigma,version=2,**kwargs):
    # name = optional(kwargs,'name','real')
    # optional_delete(kwargs,'name')
    if name == "real":
        model = load_model_real(version)
        return model
    elif name == "gray":
        model = load_model_gray(data_sigma,version)
        match_api(model)
        return model
    else:
        raise ValueError(f"Uknown model name [{name}]")

# match API of larger code-base
def match_api(model):
    fwd = model.forward
    def wrap(vid,flows=None):
        return fwd(vid,0)
    model.forward = wrap
    model.times = {}
    model.chop = False
