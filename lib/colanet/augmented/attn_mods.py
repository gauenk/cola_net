
# -- misc --
import stnls
from copy import deepcopy as dcopy
from easydict import EasyDict as edict
from ..utils import optional

# -- clean code --
from dev_basics.utils import clean_code
# from colanet.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def update_search(self,inds_is_none):
    if self.refine_inds:
        self.search_kwargs["refine_inds"] = not(inds_is_none)
        self.search = self.init_search(**self.search_kwargs)

@register_method
def init_search(self,**kwargs):
    attn_mode = optional(kwargs,"search_name","nl")
    refine_inds = optional(kwargs,"refine_inds",False)
    cfg = dcopy(kwargs)
    del cfg["attn_mode"]
    del cfg["refine_inds"]
    if "stnls" in attn_mode:
        if refine_inds: return self.init_refine(**cfg)
        else: return self.init_stnls_k(**cfg)
    # elif attn_mode == "nl":
    #     return stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
    #                                       cfg.nheads,dist_type="prod")
    # elif attn_mode == "refine":
    #     return stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
    #                                       cfg.nheads,dist_type="prod")
    elif attn_mode == "csa":
        rbounds = optional(kwargs,"reflect_bounds",False)
        stride0 = optional(kwargs,"stride0",4)
        dil = optional(kwargs,"dilation",1)
        return edict({"reflect_bounds":rbounds,"stride0":stride0,"dilation":dil})
    else:
        raise ValueError(f"Uknown attn_mode [{attn_mode}]")

@register_method
def init_refine(self,k=100,ps=7,pt=0,ws=21,wr=3,kr=1.,wt=0,
                stride0=4,stride1=1,dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):
    use_k = k > 0
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    nheads = 1
    anchor_self = True
    use_self = anchor_self
    nheads = 1
    return stnls.search.RefineSearch(ws,ps,k,wr,kr,nheads,
                                    dist_type="prod",use_adj=use_adj,
                                    anchor_self=anchor_self,
                                    dilation=dilation,rbwd=rbwd,
                                    nbwd=nbwd,exact=exact,
                                    reflect_bounds=reflect_bounds)
    # search = stnls.search_dev.init("prod_refine", k, ps, pt, ws_r, ws, nheads,
    #                               chnls=-1,dilation=dilation,
    #                               stride0=stride0, stride1=stride1,
    #                               reflect_bounds=reflect_bounds,use_k=use_k,
    #                               search_abs=search_abs,use_adj=True,
    #                               anchor_self=anchor_self,use_self=use_self,
    #                               exact=exact)
    # return search

@register_method
def init_stnls_k(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,stride0=4,stride1=1,
                dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):
    use_k = k > 0
    search_abs = ws == -1
    use_adj = True
    oh0,ow0,oh1,ow1 = 1,1,3,3
    # anchor_self = True
    anchor_self = False
    if search_abs:
        use_adj = True
        oh0,ow0,oh1,ow1 = 1,1,3,3
    full_ws = False
    fflow,bflow = None,None
    use_self = anchor_self
    nheads = 1
    return stnls.search.NonLocalSearch(ws,wt,ps,k,nheads,
                                      dist_type="prod",use_adj=use_adj,
                                      anchor_self=anchor_self,
                                      dilation=dilation,rbwd=rbwd,
                                      nbwd=nbwd,exact=exact,
                                      reflect_bounds=reflect_bounds)
                                      # oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1)
    # search = dnls.search_dev.init("prod_with_index", fflow, bflow,
    #                               k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
    #                               dilation=dilation, stride0=stride0,stride1=stride1,
    #                               reflect_bounds=reflect_bounds,use_k=use_k,
    #                               use_adj=use_adj,search_abs=search_abs,
    #                               rbwd=rbwd,nbwd=nbwd,exact=exact,full_ws=full_ws,
    #                               anchor_self=anchor_self,use_self=use_self)
    # return search

@register_method
def init_csa(self,k=100,ps=7,pt=-1,ws=-1,ws_r=-1,wt=-1,stride0=4,stride1=1,
             dilation=1,rbwd=True,nbwd=1,exact=False,
             reflect_bounds=False):
    use_k = k > 0
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    anchor_self = True
    use_self = anchor_self
    search = None#CSAWrap(oh0,oh1,ps,stride0,stride1)
    # search = stnls.search.init("prod_with_index", fflow, bflow,
    #                           k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
    #                           dilation=dilation, stride0=stride0,stride1=stride1,
    #                           reflect_bounds=reflect_bounds,use_k=use_k,
    #                           use_adj=True,search_abs=search_abs,
    #                           rbwd=rbwd,nbwd=nbwd,exact=exact,
    #                           anchor_self=anchor_self,use_self=use_self)
    return search

@register_method
def init_wpsum(self,ps=7,pt=0,dilation=1,reflect_bounds=False,
               rbwd=True,nbwd=1,exact=False,agg_fxn="unused",stride0=1):
    wpsum = stnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                dilation=dilation,
                                                reflect_bounds=reflect_bounds,
                                                adj=0, exact=exact,
                                                rbwd=rbwd,nbwd=nbwd)
    return wpsum


@register_method
def init_pdbsum(self,ps=7,pt=0,dilation=1,reflect_bounds=False,
                rbwd=True,nbwd=1,exact=False,agg_fxn="unused",stride0=1):
    pdbsum = PdbAgg(-1,ps,pt,stride0,5*1024)
    return pdbsum

@register_method
def init_agg(self,**kwargs):
    if kwargs['agg_fxn'] == "wpsum":
        return self.init_wpsum(**kwargs)
    elif kwargs['agg_fxn'] == "pdb":
        return self.init_pdbsum(**kwargs)
    raise ValueError(f"Uknown agg_fxn {kwargs['agg_fxn']}")

@register_method
def init_ifold(self,vshape,device):
    rbounds = self.search.reflect_bounds
    stride0,dil = self.search.stride0,self.search.dilation
    ifold = stnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                        adj=0,only_full=False,use_reflect=rbounds,device=device)
    return ifold

@register_method
def batching_info(self,vshape):
    B,T,C,H,W = vshape
    stride0 = self.stride0
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    ntotal = T * nH * nW
    bs = self.batchsize
    if bs == -1 or bs is None:
        nmax = 5 * (540//4) * (960//4)
        nbatch = nmax if ntotal > nmax else ntotal
    else:
        nbatch = bs
    nbatches = (ntotal-1) // nbatch + 1
    return nbatch,nbatches,ntotal
