"""
Functions for internal domain adaptation.

"""

# -- misc --
import sys,math,gc

# -- linalg --
import torch as th
import numpy as np
from einops import repeat,rearrange

# -- path mgmnt --
from pathlib import Path

# -- separate class and logic --
from colanet.utils import clean_code
from colanet.utils.misc import assert_nonan
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Testing Function for CA
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def ca_forward(self,noisy):
    model = self.model
    mod = model.head(noisy)
    nlayers = len(model.body)
    trig_layer = nlayers - (model.n_resblocks // 2 + 2)
    for lid,layer in enumerate(model.body):
        if lid == trig_layer:
            mod = layer.c1(mod)
            break
        else:
            mod = layer(mod)
    return mod
