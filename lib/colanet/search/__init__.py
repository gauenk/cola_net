"""

Interface to compare search methods

"""

# -- impl objs --
from .csa import CSASearch
from .nl import NLSearch

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_search_config = partial(extract_config,_fields)

def get_search(cfg):

    # -- unpack --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    name = optional(cfg,'name',"csa")
    k = optional(cfg,'k',50)
    ps = optional(cfg,'ps',7)
    nheads = optional(cfg,'nheads',1)
    stride0 = optional(cfg,'stride0',4)
    stride1 = optional(cfg,'stride1',1)
    ws = optional(cfg,'ws',25)
    wt = optional(cfg,'wt',0)
    if init: return

    # -- init --
    if name in ["csa","original"]:
        return CSASearch(ps,nheads,stride0,stride1)
    elif name in ["nl","ours"]:
        return NLSearch(k,ps,ws,wt,nheads,stride0,stride1)
    else:
        raise ValueError(f"Uknown search method [{name}]")

# -- fill fields --
get_search({"__init":True})
