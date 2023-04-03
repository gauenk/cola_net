
# -- misc --
import os,math,tqdm
import pprint,random,copy
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
from colanet import flow

# -- caching results --
import cache_io

# -- network --
import colanet
import colanet.configs as configs
from colanet import lightning
from colanet.utils.misc import optional,slice_flows
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils.misc import rslice,write_pickle,read_pickle
from colanet.utils.proc_utils import get_fwd_fxn
# from colanet.utils.proc_utils import spatial_chop,temporal_chop


def run_exp(_cfg):

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)

    # -- set seed --
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_deno = []
    results.mem_res = []
    results.mem_alloc = []
    results.accs = []

    # -- network --
    nchnls = 1 if cfg.bw else 3
    model = colanet.load_model(cfg)
    model.eval()
    imax = 255.
    use_chop = (cfg.attn_mode == "default") and (cfg.use_chop == "true")
    # use_chop = cfg.use_chop == "true"
    model.chop = use_chop
    print("use_chop: ",use_chop)

    # -- optional load trained weights --
    load_trained_state(model,cfg.sigma,cfg.use_train,cfg.attn_mode)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",0)
    if frame_start >= 0 and frame_end > 0:
        def fbnds(fnums,lb,ub):
            return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
                                               cfg.frame_start,cfg.frame_end)]
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data.te[index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums'].numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)
        temporal_chop = noisy.shape[0] > 20
        temporal_chop = temporal_chop and not(use_chop)

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = colanet.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 390*39#ngroups*1024

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            sigma_est = flow.est_sigma(noisy)
            flows = flow.run_batch(noisy[None,:],sigma_est)
        else:
            flows = flow.run_zeros(noisy[None,:])
        timer.stop("flow")

        # -- denoise --
        fwd_fxn = get_fwd_fxn(cfg,model)
        timer.start("deno")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = fwd_fxn(noisy/imax,flows)
        deno = deno.clamp(0.,1.)*imax
        timer.stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)

        # -- view buffer --
        accs = []
        inds = model.inds_buffer
        # kgrid = [5,25,50,75,100]
        kgrid_s = [5,50]
        kgrid_p = [50,75,100]
        for r in range(0,10,2):
            for k_s in kgrid_s:
                for k_p in kgrid_p:
                    # print(r,k)
                    accs_ = process_inds(inds,r,k_s,k_p)
                    accs_ = th.mean(accs_,2).ravel()
                    accs_ = accs_.cpu().numpy()
                    accs_ = np.r_[r,k_s,k_p,accs_]
                    accs.append(accs_)
        accs = np.stack(accs)
        # print(accs)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = colanet.utils.io.save_burst(deno,out_dir,"deno")
        # colanet.utils.io.save_burst(clean,out_dir,"clean")

        # -- psnr --
        noisy_psnrs = colanet.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        psnrs = colanet.utils.metrics.compute_psnrs(deno,clean,div=imax)
        ssims = colanet.utils.metrics.compute_ssims(deno,clean,div=imax)
        print(noisy_psnrs)
        print(psnrs)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)
        results.accs.append(accs)

    return results

def process_inds(inds,R,K_s,K_p):
    inds = inds[0] # no 2nd batching right now.
    N,B,Q,K,_ = inds.shape
    inds = inds.type(th.float32)
    accs = []
    for n0 in range(N):
        for n1 in range(n0+1,N):
            _acc = process_inds_pair(inds[n0],inds[n1],R,K_s,K_p)
            accs.append(_acc)
    accs = th.stack(accs)
    return accs

def process_inds_pair(inds0,inds1,R,K_s,K_p):
    B,Q,K,_ = inds0.shape
    inds0 = inds0.view(B*Q,K,-1)
    inds0 = inds0[:,:K_p]
    inds1 = inds1.view(B*Q,K,-1)
    inds1 = inds1[:,:K_s] # only top K_n
    p = th.inf

    if R == 0: # exact only
        dists = th.cdist(inds1,inds0,p=p)
        dists = th.any(dists<=R,-1).type(th.float32)
    else:
        # -- frames must match --
        dists = th.cdist(inds1[...,[0]],inds0[...,[0]],p=p)
        frames_eq = th.any(dists<=0,-1)

        # -- spatial distance --
        dists = th.cdist(inds1[...,1:],inds0[...,1:],p=p)
        dists = th.any(dists<=R,-1)

        # -- spatial & same frame --
        dists = th.logical_and(dists,frames_eq).type(th.float32)

    dists = th.mean(dists,-1)
    dists = dists.view(B,Q)
    return dists

def load_trained_state(model,sigma,use_train,attn_mode):

    # -- skip if needed --
    if use_train is False: return

    if attn_mode == "stnls_k":
        if np.abs(sigma-50.) < 1e-10:
            model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=38-val_loss=1.15e-03.ckpt"
        elif np.abs(sigma-30.) < 1e-10:
            model_path = "output/checkpoints/aa543914-3948-426b-b744-8403d46878cd-epoch=30.ckpt"
        elif np.abs(sigma-10.) < 1e-10:
            model_path = "output/checkpoints/b9f2e40b-9288-4800-b58b-fd94efa2c3e3-epoch=29.ckpt"
        else:
            raise ValueError("What??")
    elif attn_mode == "default":
        model_path = "output/checkpoints/dec78611-36a7-4a9e-8420-4e60fe8ea358-epoch=91-val_loss=6.63e-04.ckpt"
    else:
        raise ValueError(f"Uknown attn_mode [{attn_mode}]")

    # -- load model state --
    print("Loading Model: ",model_path)
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    if hasattr(model,"model"):
        model.model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "show_refine" # best results
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    cfg.isize = "256_256"
    # cfg.isize = "128_128"
    # cfg.isize = "none"#"128_128"
    cfg.bw = True
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.return_inds = True

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 3#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.attn_mode = "stnls_k"

    # -- get mesh --
    dnames,sigmas = ["set8"],[50]#,30.]
    # vid_names = ["tractor"]
    # vid_names = ["sunflower"]
    vid_names = ["sunflower","hypersmooth","tractor"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    ws,wt,k,bs = [29],[3],[100],[48*1024]
    flow,isizes = ["true"],["none"]
    # ca_fwd_list = ["stnls_k"]
    use_train = ["true","false"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"ws":ws,"wt":wt,
                 "isize":isizes,"use_train":use_train,#"ca_fwd":ca_fwd_list,
                 "ws":ws,"wt":wt,"k":k, "bs":bs, "use_chop":["false"]}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
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
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    # print(records.filter(like="timer"))

    # -- viz report --
    for use_train,tdf in records.groupby("use_train"):
        for ca_group,gdf in tdf.groupby("attn_mode"):
            for use_chop,cdf in gdf.groupby("use_chop"):
                for sigma,sdf in cdf.groupby("sigma"):
                    print("--- %d ---" % sigma)
                    for use_flow,fdf in sdf.groupby("flow"):
                        agg_psnrs,agg_ssims,agg_dtime = [],[],[]
                        agg_mem_res,agg_mem_alloc = [],[]
                        print("--- %s (%s,%s,%s) ---" %
                              (ca_group,use_train,use_flow,use_chop))
                        for vname,vdf in fdf.groupby("vid_name"):
                            psnrs = np.stack(vdf['psnrs'])
                            dtime = np.stack(vdf['timer_deno'])
                            mem_alloc = np.stack(vdf['mem_alloc'])
                            mem_res = np.stack(vdf['mem_res'])
                            ssims = np.stack(vdf['ssims'])
                            psnr_mean = psnrs.mean().item()
                            ssim_mean = ssims.mean().item()
                            uuid = vdf['uuid'].iloc[0]
                            # print(dtime,mem_gb)
                            # print(vname,psnr_mean,ssim_mean,uuid)
                            args = (vname,psnr_mean,ssim_mean,uuid)
                            print("%13s: %2.3f %1.3f %s" % args)
                            agg_psnrs.append(psnr_mean)
                            agg_ssims.append(ssim_mean)
                            agg_mem_res.append(mem_res.mean().item())
                            agg_mem_alloc.append(mem_alloc.mean().item())
                            agg_dtime.append(dtime.mean().item())
                        psnr_mean = np.mean(agg_psnrs)
                        ssim_mean = np.mean(agg_ssims)
                        dtime_mean = np.mean(agg_dtime)
                        mem_res_mean = np.mean(agg_mem_res)
                        mem_alloc_mean = np.mean(agg_mem_alloc)
                        uuid = gdf['uuid']
                        params = ("Ave",psnr_mean,ssim_mean,dtime_mean,
                                  mem_res_mean,mem_alloc_mean)
                        print("%13s: %2.3f %1.3f %2.3f %2.3f %2.3f" % params)


if __name__ == "__main__":
    main()
