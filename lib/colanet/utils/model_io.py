import torch as th
from pathlib import Path

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def load_checkpoint(model, path, wtype="git"):
    if wtype in ["git","original"]:
        load_checkpoint_git(model,path)
    elif wtype in ["lightning","lit"]:
        load_checkpoint_lit(model,path)
    else:
        raise ValueError(f"Uknown checkpoint weight type [{wtype}]")

def load_checkpoint_lit(model,path):
    # -- filename --
    if not Path(path).exists():
        path = str("output/checkpoints/" / Path(path))
    assert Path(path).exists()
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    model.load_state_dict(state)

def load_checkpoint_git(model,path):
    checkpoint = th.load(path)
    try:
        # model.load_state_dict(checkpoint["state_dict"])
        raise ValueError("")
    except Exception as e:
        state_dict = checkpoint["net"]
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] if 'module.' in k else k
        #     new_state_dict[name] = v
        model.load_state_dict(state_dict)
