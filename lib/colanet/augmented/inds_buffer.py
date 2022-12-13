
# -- imports --
import torch as th
from dev_basics.utils.timer import ExpTimerList

# -- separate class and logic --
from colanet.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def update_times(self,timer):
    # print(timer.names)
    if not(self.use_timer): return
    for key in timer.names:
        if key in self.times.names:
            self.times[key].append(timer[key])
        else:
            self.times[key] = [timer[key]]

@register_method
def _reset_times(self):
    self.times = ExpTimerList(self.use_timer)

@register_method
def format_inds(self,*inds_list):
    if not(self.return_inds): return None
    else: return th.stack(inds_list,0)

@register_method
def clear_inds_buffer(self):
    if not(self.use_inds_buffer): return
    self.inds_buffer = []

@register_method
def get_inds_buffer(self):
    if not(self.use_inds_buffer):
        return None
    else:
        ishape = self.inds_buffer.shape
        ishape = (ishape[0]*ishape[1],)+ishape[2:]
        return self.inds_buffer.view(ishape)

@register_method
def update_inds_buffer(self,inds):
    if not(self.use_inds_buffer): return
    if len(self.inds_buffer) == 0:
        self.inds_buffer = inds[None,:]
    else:
        self.inds_buffer = th.cat([self.inds_buffer,inds],0)
