
# -- separate class and logic --
import torch.nn as nn
from colanet.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def load_state_dict(self, state_dict, strict=True):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                if name.find('tail') == -1:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
        elif strict:
            if name.find('tail') == -1:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))

