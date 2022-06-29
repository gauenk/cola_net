
from .model import Model

def load_model(*args,**kwargs):
    # -- load model --
    model = Model(*args,**kwargs)
    return model
