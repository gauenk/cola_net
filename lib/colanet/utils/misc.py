
def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def optional_delete(pydict,key):
    if pydict is None: return
    elif key in pydict: del pydict[key]
    else: return


