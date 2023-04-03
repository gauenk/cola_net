"""

Test if our CA module matches the original CA module

"""

# -- misc --
import sys,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

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
import stnls # supporting
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import colanet
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats
import colanet.utils.metrics as metrics

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
    verbose = True
    mtype = "gray"
    ensemble = False
    chop = False
    no_grad = False

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.bw = True
    cfg.sigma = 50.

    # -- search space params --
    ws,wt = 29,0

    # -- adaptation params --
    internal_adapt_nsteps = 100
    internal_adapt_nepochs = 0

    # -- batching params --
    batch_size = -1 # unused

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    index = 0

    # -- create timer --
    timer = colanet.utils.timer.ExpTimer()

    # -- unpack --
    sample = data.te[index]
    noisy,clean = sample['noisy'][None,],sample['clean'][None,]
    noisy,clean = noisy.to(device),clean.to(device)
    h,w = 32,32
    # h,w = 64,64
    # h,w = 256,256
    clean = clean[:,:,:h,:w].contiguous()/255.
    clean = th.cat([clean,]*5,0)
    clean = clean[:,[0]].contiguous()
    noisy = clean + 25./255 * th.randn_like(clean)
    # print("noisy.shape: ",noisy.shape)
    # print("clean.shape: ",noisy.shape)
    # clean = th.randn((5,1,480,910)).to(noisy.device)
    # noisy = th.randn((5,1,480,910)).to(noisy.device)
    # noisy = th.cat([noisy,noisy],-1)
    # clean = th.cat([clean,clean],-1)
    # noisy = th.cat([noisy,noisy],-2)
    # clean = th.cat([clean,clean],-2)
    # th.autograd.set_detect_anomaly(True)

    print("noisy.shape: ",noisy.shape)

    # -- compute flow --
    flows = None

    # -- get noisy images --
    noisy_og = noisy.clone()
    # noisy_og.requires_grad_(True)
    noisy_ref = noisy.clone()
    # noisy_ref.requires_grad_(True)

    # -- original exec --
    og_model = colanet.original.load_model(mtype,sigma).eval()
    og_model.chop = chop
    timer.start("original")
    gpu_mem.reset_peak_gpu_stats()
    if no_grad:
        with th.no_grad():
            # deno_og = og_model(noisy,0,ensemble=ensemble).detach()
            deno_og = og_model.ca_forward(noisy_og).detach()
    else:
        og_model.train()
        deno_og = og_model.ca_forward(noisy_og)
    gpu_mem.print_peak_gpu_stats(True,"og",reset=True)
    # og_model.train()
    # deno_og = og_model(noisy,0,ensemble=ensemble)
    # loss = th.sum((deno_og - clean)**2).sum()
    # loss.backward()
    timer.stop("original")

    # -- each version --
    t,c,h,w = noisy.shape
    coords=[0,0,h,w]

    # -- load model --
    ref_model = colanet.refactored.load_model(mtype,sigma).eval()
    ref_model.chop = chop
    # ref_model.model.load_state_dict(og_model.model.state_dict())

    # -- optional adapt --
    run_adapt = (internal_adapt_nsteps>0) and (internal_adapt_nepochs>0)
    if run_adapt:
        ref_model.run_internal_adapt(noisy,sigma,flows=flows,
                                     ws=ws,wt=wt,batch_size=batch_size,
                                     nsteps=internal_adapt_nsteps,
                                     nepochs=internal_adapt_nepochs,
                                     verbose=True)

    # -- refactored exec --
    timer.start("refactored")
    if no_grad:
        with th.no_grad():
            # deno_ref = ref_model.my_fwd(noisy,ensemble=ensemble).detach()
            deno_ref = ref_model.ca_forward(noisy_ref).detach()
    else:
        ref_model.train()
        deno_ref = ref_model.ca_forward(noisy_ref)
    timer.stop("refactored")

    # -- viz --
    if verbose:
        print(deno_og.shape,deno_ref.shape)

    # -- test --
    error = th.mean((deno_og - deno_ref).abs()).item()
    if verbose: print("error: ",error)
    assert error < 1e-6

    # -- compute gradient --
    deno_tgt = 2.*(th.randn_like(deno_og)-0.5)
    deno_tgt = th.ones_like(deno_tgt)
    deno_tgt = th.rand_like(deno_tgt)
    loss = th.mean(th.abs(deno_og - deno_tgt))*10.
    loss.backward()
    loss = th.mean(th.abs(deno_ref - deno_tgt))*10.
    loss.backward()

    #
    # -- check grads --
    #

    # -- get grads --
    grads_og,g_og,theta_og,phi_og = get_grads(og_model,"cuda:0")
    grads_ref,g_ref,theta_ref,phi_ref = get_grads(ref_model,"cuda:0")
    rel_error = th.abs(grads_og - grads_ref)/(th.abs(grads_og)+1e-8)
    # print(len(grads_og))

    diff = th.abs(grads_og - grads_ref)
    args = th.where(diff > 1e-4)
    # almost_equal(grads_og,grads_ref)
    print(len(args[0]))
    print(grads_og[args][:5])
    print(grads_ref[args][:5])
    # exit(0)
    print(grads_og[:3])
    print(grads_ref[:3])

    if len(g_og) > 1:
        print("g")
        for i in range(3):
            print("--- %d ---" % i)
            print(g_og[i,:3])
            print(g_ref[i,:3])

            diff = (g_og[i] - g_ref[i]).abs()
            error = diff.mean().item()
            print("mean: ",error)
            error = diff.max().item()
            print("max: ",error)

    if len(theta_og) > 1:
        print("theta")
        for i in range(3):
            print("--- %d ---" % i)
            print(theta_og[i,:3])
            print(theta_ref[i,:3])

            diff = (theta_og[i] - theta_ref[i]).abs()
            error = diff.mean().item()
            print("mean: ",error)
            error = diff.max().item()
            print("max: ",error)


    if len(phi_og) > 1:
        print("phi")
        for i in range(3):
            print("--- %d ---" % i)
            print(phi_og[i,:3])
            print(phi_ref[i,:3])

            diff = (phi_og[i] - phi_ref[i]).abs()
            error = diff.mean().item()
            print("mean: ",error)
            error = diff.max().item()
            print("max: ",error)


    # -- gradient error comp --
    diff = (grads_og - grads_ref).abs()/(grads_og.abs()+1e-8)
    args = th.where(th.abs(diff) > 0.01)[0]
    print(args)
    print(grads_og[args][:10])
    print(grads_ref[args][:10])

    # -- checks --
    diff[args] = 0
    assert len(args) < 60, "allow some ineq"

    tol = 1.
    error = diff.mean().item()
    print("Mean Error: ",error)
    assert error < tol

    tol = 1e-2
    error = diff.max().item()
    print("Max Error: ",error)
    assert error < tol

    # -- viz [theta] --
    # print(theta_og.shape)
    # diff = (theta_og - theta_ref).abs()/(theta_og.abs()+1e-8)
    # args = th.where(th.abs(diff) > 0.1)
    # print(args)
    # print(theta_og[args][:10])
    # print(theta_ref[args][:10])
    # print(diff[args][:10])

    print(timer)


def almost_equal(a,b):
    tol0 = 1e-8
    tol_abs = 1e-5
    tol_rabs = 1e-4

    a_abs = a.abs()
    b_abs = b.abs()
    diff = th.abs(a - b)
    s_abs = a_abs + b_abs
    m_abs = th.min(a_abs,b_abs)

    args0 = th.where(s_abs < tol0)
    args1 = th.where(s_abs >= tol0)

    lt_abs = th.all(diff[args0] < tol_abs).item()
    lt_rabs = th.all(diff[args1]/m_abs[args1] < tol_rabs).item()
    print(diff[args1])
    print(a[args1])
    print(b[args1])
    assert lt_abs is True
    assert lt_rabs is True

def get_grads(model,device):
    grads,g,theta,phi = [],[],[],[]
    for name,param in model.named_parameters():
        if param.grad is None: continue
        if "g.weight" in name:
            print(name)
            g.append(param.grad.view(-1))
        elif "theta.weight" in name:
            print(name)
            theta.append(param.grad.view(-1))
        elif "phi.weight" in name:
            phi.append(param.grad.view(-1))
        grads.append(param.grad.view(-1))
    grads = th.round(th.cat(grads),decimals=2)
    g = th.round(th.stack(g),decimals=2) if len(g)> 0 else th.FloatTensor([0]).to(device)
    theta = th.round(th.stack(theta),decimals=2) if len(theta) > 0 else th.FloatTensor([0]).to(device)
    phi = th.round(th.stack(phi),decimals=2) if len(phi) > 0 else th.FloatTensor([0]).to(device)
    return grads,g,theta,phi
