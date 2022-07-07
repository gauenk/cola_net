"""

Create an error map of the backward step through colanet
between the exact and approximate gradient cuda kernels

"""


# -- misc --
import os,math,tqdm
import pprint,copy,random
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import colanet
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils.timer import ExpTimer
from colanet.utils.metrics import compute_psnrs,compute_ssims
from colanet.utils.misc import rslice,write_pickle,read_pickle


class WrapColaNet(th.nn.Module):

    def __init__(self,mtype,sigma,flow=True,ensemble=False,
                 ca_fwd="dnls_k",isize=None,exact=False,device="cuda"):
        super().__init__()
        self.mtype = mtype
        self.sigma = sigma
        self._model = [colanet.refactored.load_model(mtype,sigma)]
        self.net = self._model[0].model
        self.net.body[8].ca_forward_type = ca_fwd
        self.net.body[8].exact = exact
        self.flow = flow
        self.isize = isize
        self.device = device

    def forward(self,vid):
        flows = self._get_flow(vid)
        if not(self.isize is None):
            deno = self.net(vid,flows=flows,region=None)
        else:
            deno = self.forward_full(vid,flows)
        deno = th.clamp(deno,0.,1.)
        return deno

    def forward_full(self,vid,flows):
        model = self._model[0]
        model.model = self.net
        deno = model.forward_chop(vid,flows=flows)
        return deno

    def _get_flow(self,vid):
        if self.flow == True:
            noisy_np = vid.detach().cpu().numpy()
            if noisy_np.shape[1] == 1:
                noisy_np = np.repeat(noisy_np,3,axis=1)
            flows = svnlb.compute_flow(noisy_np,self.sigma)
            flows = edict({k:th.from_numpy(v).to(self.device) for k,v in flows.items()})
        else:
            t,c,h,w = vid.shape
            zflows = th.zeros((t,2,h,w)).to(self.device)
            flows = edict()
            flows.fflow,flows.bflow = zflows,zflows
        return flows

def run_experiment(cfg):

    # -- set seed --
    random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # -- init log dir --
    log_dir = Path(cfg.log_root) / str(cfg.uuid)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # -- network --
    model = WrapColaNet(cfg.mtype,cfg.sigma,
                        cfg.flow=="true",cfg.ensemble=="true",
                        cfg.ca_fwd,cfg.isize,cfg.exact == "true")
    model = model.to(cfg.device)

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    sample = data.val[0]
    index,region = sample['index'].item(),sample['region']
    noisy = rslice(sample['noisy'].to(cfg.device),region)
    clean = rslice(sample['clean'].to(cfg.device),region)

    # -- set autograd --
    model.eval()
    noisy.requires_grad_(True)

    # -- forward --
    deno = model(noisy)

    # -- backward --
    loss = th.mean((deno - clean)**2)
    loss.backward()

    # -- psnr reference --
    psnr = np.mean(compute_psnrs(deno,clean,div=1.))

    # -- grad --
    grad = noisy.grad

    # -- save grad to dir --
    grad_root = Path("./output/bwd_error_map/grads/")
    path = "exact" if cfg.exact == "true" else "not_exact"
    file_stem = "%d_%d.pt" % (cfg.seed,cfg.rep_id)
    grad_dir = grad_root / path
    if not grad_dir.exists(): grad_dir.mkdir(parents=True)
    grad_fn = str(grad_dir / file_stem)
    th.save(grad.cpu(),grad_fn)

    # -- results --
    results = edict()
    results.grad = grad_fn
    results.loss = loss.item()
    results.psnr = psnr.item()
    results.vid_index = index
    return results

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/colanet/output/checkpoints/"
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.batch_size = 1
    cfg.saved_dir = "./output/saved_results/"
    cfg.device = "cuda:0"
    cfg.dname = "davis"
    cfg.flow = "true"
    cfg.mtype = "gray"
    cfg.bw = True
    cfg.nsamples_at_testing = 10
    cfg.nsamples_tr = 500
    cfg.nsamples_val = 30
    cfg.rand_order_val = False
    cfg.index_skip_val = 5
    cfg.nepochs = 5
    cfg.ensemble = "false"
    cfg.log_root = "./output/bwd_error_map/log"
    return cfg

def main():
    # -- print os pid --
    print("PID: ",os.getpid())

    # -- init --
    verbose = True
    cache_name = "bwd_error_map"
    cache = cache_io.ExpCache(".cache_io",cache_name)
    # cache.clear()

    # -- create exp list --
    ws,wt = [10],[5]
    sigmas = [30.]
    isizes = ["96_96"]
    exact = ["false"]
    ca_fwd_list = ["dnls_k"]
    rep_ids = list(np.arange(3))
    seeds = list(np.arange(100))
    exp_lists = {"sigma":sigmas,"ws":ws,"wt":wt,"isize":isizes,
                 "ca_fwd":ca_fwd_list,"exact":exact,"seed":seeds,
                 "rep_id":rep_ids}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    nexps = len(exps)

    # -- group with default --
    cfg = default_cfg()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- launch each experiment --
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_experiment(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- results --
    records = cache.load_flat_records(exps)
    print(records)

    # -- compare to gt --
    run_exp(


if __name__ == "__main__":
    main()

