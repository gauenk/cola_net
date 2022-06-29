
def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default
