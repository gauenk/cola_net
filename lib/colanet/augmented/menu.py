
"""

The number of layers ~= 50
This is too many params to set by hand.
Instead, fix the list of parameters using a menu.

"""

import copy
dcopy = copy.deepcopy
import numpy as np
from easydict import EasyDict as edict
from ..utils import optional

def fill_menu(_cfgs,fields,menu_cfgs):
    """

    Fill the input "_cfgs" fields using a menu.

    """

    # -- fields to fill --
    mfields = {"attn":[],
               "search":["search_name","use_state_update"],
               "normz":[],"agg":[],}

    # -- filling --
    cfgs = []
    for menu_cfg in menu_cfgs:
        cfgs_m = edict()
        for field in fields:
            cfg_f = dcopy(_cfgs[field])
            for fill_key in mfields[field]:
                cfg_f[fill_key] = menu_cfg[fill_key]
            cfgs_m[field] = cfg_f
        cfgs.append(cfgs_m)
    return cfgs

def extract_menu_cfg_impl(cfg,depth):
    # -- unpack attn name --
    # ...

    # -- unpack search name --
    # "search_vX" in ["exact","refine","approx_t","approx_s","approx_st"]
    p = edict()
    pairs = {"search_menu_name":"full","search_v0":"exact","search_v1":"exact"}
    for key,val in pairs.items():
        p[key] = optional(cfg,key,val)
    search_names = search_menu(p.search_menu_name,depth,
                               p.search_v0,p.search_v1)

    # -- search params from names --
    nblocks = len(search_names)
    params = search_params_from_names(search_names,nblocks)

    # -- unpack normz name --
    # ...

    # -- unpack agg name --
    # ...

    # -- grouped pairs --
    pairs = {"search_name":search_names,"use_state_update":params.use_state_updates}
    L = len(search_names)

    # -- format return value; a list of pydicts --
    blocks = []
    for l in range(L):
        block_l = edict()
        for key,val_list in pairs.items():
            block_l[key] = val_list[l]
        blocks.append(block_l)

    return blocks

def search_params_from_names(search_names,nblocks):
    """
    Create derived parameters from parsed parameters

    """

    # -- init --
    params = edict()
    params.use_state_updates = []

    # -- fill --
    for i in range(nblocks):
        any_refine = np.any(np.array(search_names)=="refine")
        params.use_state_updates.append(any_refine)
    return params

def search_menu(menu_name,depth,v0,v1):
    nblocks = 2*np.sum(depth[:-1]) + depth[-1]
    nblocks = int(nblocks)

    if menu_name == "full":
        return [v0,]*nblocks
    elif menu_name == "one":
        return [v0,] + [v1,]*(nblocks-1)
    elif menu_name == "first":
        names = []
        for depth_i in depth:
            names_i = [v0,] + [v1,]*(depth_i-1)
            names.extend(names_i)
        for depth_i in reversed(depth[:-1]):
            names_i = [v0,] + [v1,]*(depth_i-1)
            names.extend(names_i)
        return names
    elif menu_name == "nth":
        names = []
        for i in range(nblocks):
            if (i % menu_n == 0) or i == 0:
                names.append(v0)
            else:
                names.append(v1)
        return names
    else:
        raise ValeError("Uknown search type in menu [%s]" % menu_name)

