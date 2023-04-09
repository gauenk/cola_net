
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

import torch as th

from .grecc_rcaa import RR as ColaNet
from ..utils import optional as _optional

from colanet.utils import model_io
from dev_basics import arch_io
from .menu import extract_menu_cfg_impl,fill_menu

# -- auto populate fields to extract config --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init static variable
extract_config = econfig.extract_config # rename extraction

# _fields = []
# def optional_full(init,pydict,field,default):
#     if not(field in _fields) and init:
#         _fields.append(field)
#     return _optional(pydict,field,default)

@econfig.set_init
def load_model(cfg):

    # -- allows for all keys to be aggregated at init --
    econfig.set_cfg(cfg)
    # init = _optional(cfg,'__init',False) # purposefully weird key
    # optional = partial(optional_full,init)

    # -- unpack configs --
    depth = [3]
    pairs = {"io":io_pairs(),
             "arch":arch_pairs(),
             "search":search_pairs(),}
    menu_cfgs = extract_menu_cfg(cfg,depth)
    device = econfig.optional(cfg,"device","cuda:0")
    cfgs = econfig.extract_set(pairs)
    if econfig.is_init: return
    # print(menu_cfgs)

    # -- fill blocks with menu --
    # fields = ["attn","search","normz","agg"]
    fields = ["search"]
    blocks = fill_menu(cfgs,fields,menu_cfgs)
    # print(blocks)

    # -- init model --
    model = ColaNet(cfgs.arch,blocks)#search_cfg)

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- device --
    model = model.to(device)

    return model

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        cfg.pretrained_root = cfg.pretrained_root.replace("aaai23","icml23")
        if cfg.pretrained_type == "git":
            model_io.load_checkpoint(model,cfg.pretrained_path,
                                    cfg.pretrained_root,cfg.pretrained_type)
        else:
            arch_io.load_checkpoint(model,cfg.pretrained_path,
                                    cfg.pretrained_root,cfg.pretrained_type)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_pairs(pairs,_cfg,optional):
    cfg = edict()
    for key,val in pairs.items():
        cfg[key] = optional(_cfg,key,val)
    return cfg

def io_pairs():
    # base = Path("weights/checkpoints/DN_Gray/res_cola_v2_6_3_%d_l4/" % sigma)
    # pretrained_path = base / "model/model_best.pt"
    base = ""
    pretrained_path = ""
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs
    # return extract_pairs(pairs,_cfg,optional)

def search_pairs():
    pairs = {
        "ps":7,"ws":21,"wt":0,
        "wr":3,"kr":1.,"k_s":100,"k_a":100,
        "stride0":4,"stride1":1,"batchsize":-1,"pt":1,
        "rbwd":False,"nbwd":1,"exact":False,
        "reflect_bounds":False,
        "refine_inds":[False,False,False],
        "dilation":1,"return_inds":False,
        "softmax_scale":10,
        "attn_timer":False,"anchor_self":True}
    return pairs
    # return extract_pairs(pairs,_cfg,optional)

def arch_pairs():
    pairs = {"scale":[1],"self_ensemble":False,
             "chop":False,"precision":"single",
             "cpu":False,"n_GPUs":1,"pre_train":".",
             "save_models":False,"model":"COLA","mode":"E",
             "print_model":False,"resume":0,"seed":1,
             "n_resblock":16,"n_feats":64,"n_colors":1,
             "res_scale":1,"rgb_range":1.,"stages":6,
             "blocks":3,"act":"relu","sigma":0.,
             "arch_return_inds":False,"device":"cuda:0",
             "attn_timer":False,"add_SE":False}
    return pairs
    # return extract_pairs(pairs,_cfg,optional)

def extract_io_config(_cfg,optional):
    pairs = io_pairs()
    return extract_pairs(pairs,_cfg,optional)

def extract_arch_config(_cfg,optional):
    pairs = arch_pairs()
    return extract_pairs(pairs,_cfg,optional)

def extract_search_config(_cfg,optional):
    pairs = search_pairs()
    return extract_pairs(pairs,_cfg,optional)

def extract_menu_cfg(_cfg,depth):

    """

    Extract unique values for each _block_
    This can get to sizes ~=50
    So a menu is used to simplify setting each of the 50 parameters.
    These "fill" the fixed configs above.

    """

    cfg = econfig.extract_pairs({'search_menu_name':'full',
                                 "search_v0":"exact",
                                 "search_v1":"refine"},_cfg)
    return extract_menu_cfg_impl(cfg,depth)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Extracting Relevant Fields from Larger Dict
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# def extract_model_config(cfg):
#     # -- auto populated fields --
#     fields = _fields
#     model_cfg = {}
#     for field in fields:
#         if field in cfg:
#             model_cfg[field] = cfg[field]
#     return edict(model_cfg)

# # -- run to populate "_fields" --
# load_model(edict({"__init":True}))


