"""

Test versions of Colanet to differences in output due to code modifications.

"""

# -- misc --
import sys,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- optical flow --
import svnlb

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import dnls # supporting
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import colanet
from colanet.utils.misc import rslice
import colanet.utils.metrics as metrics
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils.metrics import compute_psnrs,compute_ssims
from colanet.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_denose_rgb/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.use_deterministic_algorithms(True)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    test_lists = {"sigma":[50.]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# -->  Test original vs refactored code base  <--
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# @pytest.mark.skip()
def test_original_refactored(sigma):

    # -- params --
    device = "cuda:0"
    vid_set = "bsd68"
    vid_set = "set8"
    verbose = True
    mtype = "gray"
    ensemble = False
    chop = False
    # ca_fwd = "dnls_k"
    ca_fwd = "default"

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = "motorbike"
    cfg.bw = True
    cfg.sigma = 50.

    # -- search space params --
    ws,wt = 29,0

    # -- adaptation params --
    internal_adapt_nsteps = 500
    internal_adapt_nepochs = 1

    # -- batching params --
    batch_size = -1 # unused

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    if "vid_name" in cfg:
        indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
        index = indices[0]
    else: index = 0

    # -- create timer --
    timer = colanet.utils.timer.ExpTimer()

    # -- unpack --
    sample = data.te[index]
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = noisy.to(device),clean.to(device)
    # t,h,w = 4,128,128
    t,h,w = 4,256,256
    noisy = noisy[:t,:,:h,:w].contiguous()
    clean = clean[:t,:,:h,:w].contiguous()
    # noisy = noisy[:,:,:128,:128].contiguous()
    # clean = clean[:,:,:128,:128].contiguous()
    noisy /= 255.
    clean /= 255.
    noisy = noisy[:,[0]].contiguous()
    clean = clean[:,[0]].contiguous()
    print("noisy.shape: ",noisy.shape)
    print("clean.shape: ",noisy.shape)
    # noisy = th.cat([noisy,noisy])
    # clean = th.cat([clean,clean])
    # noisy = th.cat([noisy,noisy],-1)
    # clean = th.cat([clean,clean],-1)
    # noisy = th.cat([noisy,noisy],-2)
    # clean = th.cat([clean,clean],-2)
    dnls.testing.data.save_burst(noisy,"./output","noisy")
    dnls.testing.data.save_burst(clean,"./output","clean")
    print("noisy.shape: ",noisy.shape)

    # -- compute flow --
    # flows = _get_flow(noisy,cfg.sigma)
    flows = _get_flow(clean,0.)

    # -- get noisy images --
    noisy_og = noisy.clone()
    noisy_og.requires_grad_(True)
    noisy_ref = noisy.clone()
    noisy_ref.requires_grad_(True)

    # -- original exec --
    og_model = colanet.original.load_model(mtype,sigma).eval()
    og_model.chop = chop
    og_model.eval()

    gpu_mem.reset_peak_gpu_stats()
    timer.start("original")
    with th.no_grad():
        deno_og = og_model(noisy_og,0,ensemble=ensemble).detach()
    # og_model.train()
    # deno_og = og_model(noisy,0,ensemble=ensemble)
    # loss = th.sum((deno_og - clean)**2).sum()
    # loss.backward()
    timer.stop("original")
    gpu_mem.print_peak_gpu_stats(True,"og",reset=True)


    # -- each version --
    t,c,h,w = noisy.shape
    coords=[0,0,h,w]
    for ref_version in ["ref"]: #["original","ref"]:

        # -- load model --
        ref_model = colanet.refactored.load_model(mtype,sigma).eval()
        ref_model.chop = chop
        ref_model.model.body[8].ca_forward_type = ca_fwd


        # -- optional adapt --
        run_adapt = (internal_adapt_nsteps>0) and (internal_adapt_nepochs>0)
        if run_adapt:
            # ref_model.run_internal_adapt(noisy,sigma,flows=flows,
            #                              ws=ws,wt=wt,batch_size=batch_size,
            #                              nsteps=internal_adapt_nsteps,
            #                              nepochs=internal_adapt_nepochs,
            #                              clean_gt=clean,verbose=True)
            ref_model.run_external_adapt(clean,sigma,flows=flows,
                                         ws=ws,wt=wt,batch_size=batch_size,
                                         nsteps=internal_adapt_nsteps,
                                         nepochs=internal_adapt_nepochs,
                                         noisy_gt=noisy,verbose=True)
            ref_model.eval()

        # -- refactored exec --
        timer.start("refactored")
        with th.no_grad():
            deno_ref = ref_model(noisy_ref,0,ensemble=ensemble,flows=flows).detach()
            # deno_ref = ref_model.my_fwd(noisy,ensemble=ensemble).detach()
        gpu_mem.print_peak_gpu_stats(True,"og",reset=True)
        timer.stop("refactored")

        # -- viz --
        if verbose:
            print(deno_og.shape,clean.shape)
            print("og: ",metrics.compute_psnrs(deno_og,clean,1.))
            print("ref: ",metrics.compute_psnrs(deno_ref,clean,1.))
            print(timer)

        # -- test --
        error = th.sum((deno_og - deno_ref)**2).item()
        if verbose: print("error: ",error)
        assert error < 1e-15

def _get_flow(vid,sigma):
    device = vid.device
    noisy_np = vid.cpu().numpy()
    if noisy_np.shape[1] == 1:
        noisy_np = np.repeat(noisy_np,3,axis=1)
    flows = svnlb.compute_flow(noisy_np,sigma)
    flows = edict({k:th.from_numpy(v).to(device) for k,v in flows.items()})
    return flows

@pytest.mark.skip("a feature that wont work for now.")
def test_region(sigma):
    """
    This feature won't work because the model cascades multiple layers
    of non-local modules, so the entire search space must continue
    to be computed; only 1 of 3 layers can be made more efficient.

    Additionally, even if all 3 can be computed efficiently
    the output won't be equal since the batch-norm layers layer
    change the final output.

    """

    # -- params --
    device = "cuda:0"
    vid_set = "bsd68"
    vid_set = "set8"
    verbose = True
    mtype = "gray"
    ensemble = False
    chop = False
    region_tmp = "2_96_96"
    region = [0,2,32,32,96,96]

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = "motorbike"
    cfg.bw = True
    cfg.sigma = 50.

    # -- search space params --
    ws,wt = 29,0

    # -- adaptation params --
    internal_adapt_nsteps = 500
    internal_adapt_nepochs = 0

    # -- batching params --
    batch_size = -1 # unused

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    if "vid_name" in cfg:
        indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
        index = indices[0]
    else: index = 0

    # -- create timer --
    timer = colanet.utils.timer.ExpTimer()

    # -- unpack --
    sample = data.te[index]
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = noisy.to(device),clean.to(device)
    t,h,w = 4,128,128
    noisy = noisy[:t,[0],:h,:w].contiguous()/255.
    clean = clean[:t,[0],:h,:w].contiguous()/255.
    print("noisy.shape: ",noisy.shape)
    print("clean.shape: ",noisy.shape)
    # dnls.testing.data.save_burst(noisy,"./output","noisy")
    # dnls.testing.data.save_burst(clean,"./output","clean")

    # -- compute flow --
    flows = None

    # -- get noisy images --
    noisy_full = noisy.clone()
    noisy_full.requires_grad_(True)
    noisy_region = noisy.clone()
    noisy_region.requires_grad_(True)

    # -- load model --
    model = colanet.refactored.load_model(mtype,sigma).eval()
    model.chop = chop

    #
    #
    # -- meat starts below --
    #
    #

    # -- run full --
    timer.start("full")
    gpu_mem.print_peak_gpu_stats(False,"full",reset=True)
    deno_full = model(noisy_full,0,ensemble=ensemble,region=None)
    gpu_mem.print_peak_gpu_stats(True,"full",reset=True)
    timer.stop("full")

    # -- run region --
    timer.start("region")
    gpu_mem.print_peak_gpu_stats(False,"region",reset=True)
    deno_region = model(noisy_region,0,ensemble=ensemble,region=region)
    gpu_mem.print_peak_gpu_stats(True,"region",reset=True)
    timer.stop("region")

    # -- cropped --
    clean_region = rslice(clean,region)
    df_region = rslice(deno_full,region) # deno_full_region
    print("clean_region.shape: ",clean_region.shape)
    print("df_region.shape: ",df_region.shape)
    print("deno_region.shape: ",deno_region.shape)

    # -- viz --
    if verbose:
        print("full: ",metrics.compute_psnrs(df_region,clean_region,1.))
        print("region: ",metrics.compute_psnrs(deno_region,clean_region,1.))
        print(timer)

    # -- test --
    error = th.sum((df_region - deno_region)**2).item()
    if verbose: print("error: ",error)
    assert error < 1e-15

