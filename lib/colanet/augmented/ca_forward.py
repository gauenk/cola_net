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
    mod = self.head(noisy)
    nlayers = len(self.body)
    trig_layer = nlayers - (self.n_resblocks // 2 + 2)
    for lid,layer in enumerate(self.body):
        if lid == trig_layer:
            mod = layer.c1.CAUnit(mod,None)[0]
            break
        else:
            mod = layer(mod)
    return mod

