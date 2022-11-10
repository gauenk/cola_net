try:
    from .merge_unit import merge_block
    from .DnCNN_Block import DnCNN
except:
    from model.merge_unit import merge_block
    from model.DnCNN_Block import DnCNN

import torch.nn as nn
import torch

class MergeNet(nn.Module):
    def __init__(self,in_channels,intermediate_channels,vector_length,
                 use_multiple_size,dncnn_depth,num_merge_block,use_topk=False):
        super(MergeNet,self).__init__()
        layers = []
        for i in range(num_merge_block):
            if i == 0:
                layer_i = DnCNN(nplanes_in=in_channels,
                                nplanes_out=intermediate_channels,
                                features=intermediate_channels,kernel=3,
                                depth=dncnn_depth)
                self.add_module('dncnn{}'.format(i), layer_i)
                layer_i = merge_block(in_channels=intermediate_channels,
                                      out_channels=intermediate_channels,
                                      vector_length=vector_length,
                                      use_multiple_size=use_multiple_size,
                                      use_topk=use_topk)
                self.add_module('merge{}'.format(i), layer_i)
            else:
                layer_i = DnCNN(nplanes_in=intermediate_channels,
                                nplanes_out=intermediate_channels,
                                features=intermediate_channels,
                                kernel=3,depth=dncnn_depth)
                self.add_module('dncnn{}'.format(i), layer_i)

                layer_i = merge_block(in_channels=intermediate_channels,
                                      out_channels=intermediate_channels,
                                      vector_length=vector_length,
                                      use_multiple_size=use_multiple_size,
                                      use_topk=use_topk)
                self.add_module('merge{}'.format(i), layer_i)
        layer_i = DnCNN(nplanes_in=intermediate_channels, nplanes_out=in_channels,
                        features=intermediate_channels,kernel=3, depth=dncnn_depth)
        self.add_module('dncnn{}'.format(i), layer_i)
        # self.model = nn.Sequential(*layers)

    def forward(self, x):
        coords=None
        x_i = x
        for name, module in self._modules.items():
            if 'merge' in name:
                x_i = module(x_i)#,coords)
            else:
                x_i = module(x_i)
        return x+out

