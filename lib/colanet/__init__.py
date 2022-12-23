
# -- code api --
from . import original
from . import refactored
from . import batched
from . import configs
from . import lightning
from . import flow
from . import augmented
from .augmented import extract_model_config

# -- publication api --
from . import aaai23
from . import icml23

# -- api for searching --
from . import search
from .search import get_search,extract_search_config

# -- model api --
from .utils import optional

def load_model(cfg):
    mtype = optional(cfg,'model_type','augmented')
    if mtype == "augmented":
        return augmented.load_model(cfg)
    elif mtype == "refactored":
        nchnls = 1
        name = "gray"
        return refactored.load_model(cfg,name,2,nchnls)
    elif mtype == "original":
        nchnls = 1
        name = "gray"
        sigma = optional(cfg,'sigma',30)
        return original.load_model(name,sigma,2)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
