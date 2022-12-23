import torch as th
from pathlib import Path

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def resolve_path(path,root):
    if not Path(path).exists():
        path_ = Path(root) / Path(path)
        if not(path_.exists()):
            path_ = Path(root) / "output/checkpoints/" / Path(path)
        path = path_
    assert Path(path).exists()
    return str(path)

def load_checkpoint(model, path, root, wtype="git"):
    full_path = resolve_path(path,root)
    if wtype in ["git","original"]:
        load_checkpoint_git(model,full_path)
    elif wtype in ["lightning","lit"]:
        load_checkpoint_lit(model,full_path)
    else:
        raise ValueError(f"Uknown checkpoint weight type [{wtype}]")

def load_checkpoint_lit(model,path):
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    model.load_state_dict(state)

def load_checkpoint_git(model,path):
    # -- filename --
    checkpoint = th.load(path)
    model.load_state_dict(checkpoint)
    # try:
    #     # model.load_state_dict(checkpoint["state_dict"])
    #     raise ValueError("")
    # except Exception as e:
    #     state_dict = checkpoint["net"]
    #     # new_state_dict = OrderedDict()
    #     # for k, v in state_dict.items():
    #     #     name = k[7:] if 'module.' in k else k
    #     #     new_state_dict[name] = v
    #     model.load_state_dict(state_dict)
