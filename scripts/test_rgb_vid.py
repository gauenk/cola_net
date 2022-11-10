
# -- misc --
import os,math,tqdm
import pprint,random
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
# import svnlb
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
from colanet.utils.proc_utils import get_fwd_fxn#spatial_chop,temporal_chop


def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.adapt_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []
    results.mem_res = []
    results.mem_alloc = []

    # -- network --
    nchnls = 1 if cfg.bw else 3
    model = colanet.load_model(cfg)
    model.eval()
    imax = 255.
    use_chop = (cfg.ca_fwd == "default") and (cfg.use_chop == "true")
    print("use_chop: ",use_chop)
    model.chop = use_chop

    # -- optional load trained weights --
    load_trained_state(model,cfg.use_train,cfg.ca_fwd,cfg.sigma,cfg.ws,cfg.wt)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    # indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    indices = data_hub.filter_subseq(data.te,cfg.vid_name,cfg.frame_start,cfg.frame_end)

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
        timer.sync_stop("flow")

        # -- internal adaptation --
        timer.start("adapt")
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        adapt_psnrs = [0.]
        if run_internal_adapt:
            adapt_psnrs = model.run_internal_adapt(
                noisy,cfg.sigma,flows=flows,
                ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                nsteps=cfg.internal_adapt_nsteps,
                nepochs=cfg.internal_adapt_nepochs,
                sample_mtype=cfg.adapt_mtype,
                clean_gt = clean,
                region_gt = [2,4,128,256,256,384]
            )
        timer.sync_stop("adapt")

        # -- denoise --
        fwd_fxn = get_fwd_fxn(cfg,model)
        with th.no_grad():
            deno = fwd_fxn(noisy/imax,flows)
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        timer.sync_start("deno")
        with th.no_grad():
            deno = fwd_fxn(noisy/imax,flows)
        deno = deno.clamp(0.,1.)*imax
        timer.sync_stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)

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
        results.adapt_psnrs.append(adapt_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)
        print(timer)

    return results

def load_trained_state(model,use_train,ca_fwd,sigma,ws,wt):

    # -- skip if needed --
    if not(use_train == "true"): return

    # -- open training cache info --
    cache_dir = ".cache_io"
    cache_name = "train_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- create config --
    cfg = configs.default_train_cfg()
    cfg.bw = True
    cfg.ws = ws
    cfg.wt = wt
    cfg.sigma = sigma
    cfg.isize = "128_128" # a fixed training parameters
    cfg.ca_fwd = ca_fwd

    # -- read cache --
    results = cache.load_exp(cfg) # possibly load result
    if ca_fwd == "dnls_k":
        # model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=99.ckpt"
        # model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=81-val_loss=1.24e-03.ckpt"
        if np.abs(sigma-50.) < 1e-10:
            model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=38-val_loss=1.15e-03.ckpt"
        elif np.abs(sigma-30.) < 1e-10:
            model_path = "output/checkpoints/aa543914-3948-426b-b744-8403d46878cd-epoch=30.ckpt"
        elif np.abs(sigma-10.) < 1e-10:
            model_path = "output/checkpoints/b9f2e40b-9288-4800-b58b-fd94efa2c3e3-epoch=29.ckpt"
        else:
            raise ValueError("What??")
    elif ca_fwd == "default":
        model_path = "output/checkpoints/dec78611-36a7-4a9e-8420-4e60fe8ea358-epoch=91-val_loss=6.63e-04.ckpt"
    else:
        raise ValueError(f"Uknown ca_fwd [{ca_fwd}]")

    # -- load model state --
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    if hasattr(model,"model"):
        model.model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model

def save_path_from_cfg(cfg):
    path = Path(cfg.dname) / cfg.vid_name
    train_str = "train" if  cfg.train == "true" else "notrain"
    path = path / "%s_%s" % (cfg.ca_fwd,train_str)

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    # cache_name = "test_rgb_net" # best results (aaai23)
    cache_name = "test_rgb_net_cvpr23" # best results
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    # cfg.isize = "256_256"
    cfg.isize = "128_128"
    # cfg.isize = "none"#"128_128"
    cfg.bw = True
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.attn_mode = "dnls_k"

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 3#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames


    # -- get mesh --
    dnames,sigmas = ["set8"],[30]#,30.]
    # vid_names = ["tractor"]
    # vid_names = ["sunflower"]
    # vid_names = ["sunflower","hypersmooth","tractor"]
    vid_names = ["snowboard","sunflower","tractor","motorbike",
                 "hypersmooth","park_joy","rafting","touchdown"]
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]

    # -- standard --
    # ws,wt = [27],[3]
    # cfg.k_s = 100
    # cfg.k_a = 100
    # cfg.refine_inds = "f-f-f"

    # -- prop0 --
    ws,wt = ['27-3-3'],['3-0-0']
    cfg.k_s = 100
    cfg.k_a = 100
    cfg.ws = 27
    cfg.wt = 3
    cfg.ws_r = 1
    # cfg.k_a = 50
    # cfg.refine_inds = "f-f-f"

    # -- prop1 --
    # ws,wt = ['27-27-3'],['3-3-0']
    # cfg.k_s = '100-500-100'
    # cfg.k_a = '100-100-100'
    # cfg.k_a = 50
    # cfg.refine_inds = "f-f-t"

    # -- prop2 --
    # ws,wt = ['27-27-27'],['3-3-3']
    # cfg.k_s = '100-100-100'
    # cfg.k_a = '100-100-100'
    # cfg.refine_inds = "f-f-f"

    k,sb = [100],[48*1024]#1024*1]
    flow,isizes,adapt_mtypes = ["true"],["none"],["rand"]
    ca_fwd_list,use_train = ["dnls_k"],["true"]
    refine_inds = ["f-f-f","f-f-t","f-t-f","f-t-t"]
    model_type = ['augmented']
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"ca_fwd":ca_fwd_list,
                 "ws":ws,"wt":wt,"k":k, "sb":sb, "use_chop":["false"],
                 "model_type":model_type,"refine_inds":refine_inds}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two

    # cfg.attn_mode = "dnls_k"
    # cfg.ws = "27-5-5"
    # cfg.wt = "3-0-0"
    # cfg.k_s = '500-500-100'
    # cfg.k_a = 100
    # cfg.refine_inds = "f-t-t"
    # exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    # cache_io.append_configs(exps_a,cfg) # merge the two

    # -- original w/out training --
    cfg.ws = 27
    exp_lists['model_type'] = ['refactored']
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exp_lists['ca_fwd'] = ["default"]
    exp_lists['use_chop'] = ["false"]
    exp_lists['sb'] = [1]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.bw = True
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a# + exps_b

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

        clear_exp = exp.attn_mode == "dnls_k" and exp.model_type == "refactored"
        clear_exp = clear_exp and (exp.ws != 27)
        clear_exp = clear_exp or ('t' in exp.refine_inds)
        # if clear_exp:
        #     cache.clear_exp(uuid)
        # if exp.use_chop == "false" and exp.ca_fwd != "dnls_k":
        #     cache.clear_exp(uuid)
        # if exp.ca_fwd != "dnls_k" and exp.sigma == 30.:
        #     cache.clear_exp(uuid)
        # if exp.sigma == 30. and exp.ca_fwd == "dnls_k":
        #     cache.clear_exp(uuid)
        # if exp.sigma == 10. and exp.ca_fwd == "dnls_k":
        #     cache.clear_exp(uuid)
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
        for ca_group,gdf in tdf.groupby("refine_inds"):
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
