
def config_to_list(search_cfg,nblocks=3):

    # -- config list --
    cfgs = []
    for _ in range(nblocks):
        cfgs.append(edict())

    # -- transpose --
    for key,val in search_cfg.items():
        if isinstance(val,list):
            assert len(val) == nblocks
            for i,v in enumerate(val):
                cfgs[i][key] = v
        elif isinstance(val,str):
            # -- expand --
            val = val.split("-")
            if len(val) == 1:
                val = [val[0],] * nblocks
            else:
                assert len(val) == nblocks
            for i,v in enumerate(val):
                cfgs[i][key] = translate(key,v)
        else:
            for i in range(nblocks):
                cfgs[i][key] = val
    return cfgs

def translate(key,val):
    if key == "field":
        return int(val)
    elif key == "attn_mode":
        return val
    elif key in ["refine_inds"]:
        return val == "t" # true/false via "t" or "f"
    else:
        return int(val)
