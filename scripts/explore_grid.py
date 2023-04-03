"""
This is identical to "test_rgb_vid"
but using a different grid and cache.

"""


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
from colanet import lightning
import colanet.configs as configs
import colanet.explore_configs as explore_configs
from colanet.utils.misc import optional,slice_flows
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils.misc import rslice,write_pickle,read_pickle

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
    results.flops = []

    # -- network --
    nchnls = 1 if cfg.bw else 3
    model = colanet.refactored.load_model(cfg.mtype,cfg.sigma,2,nchnls).to(cfg.device)
    model.eval()
    imax = 255.
    model.model.body[8].ca_forward_type = cfg.ca_fwd
    model.model.body[8].ws = cfg.ws
    model.model.body[8].wt = cfg.wt
    model.model.body[8].k = cfg.k
    model.model.body[8].sb = cfg.sb
    use_chop = (cfg.ca_fwd == "default") and (cfg.use_chop == "true")
    model.chop = use_chop

    # -- optional load trained weights --
    load_trained_state(model,cfg.use_train,cfg.ca_fwd,cfg.sigma,cfg.ws,cfg.wt)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",0)
    if frame_start >= 0 and frame_end > 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
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
        vid_frames = sample['fnums'].cpu().numpy()
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
            est_sigma = flow.est_sigma(noisy)
            flows = flow.run(noisy,est_sigma)
            # noisy_np = noisy.cpu().numpy()
            # if noisy_np.shape[1] == 1:
            #     noisy_np = np.repeat(noisy_np,3,axis=1)
            # flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            # flows=edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

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
        timer.stop("adapt")

        # -- denoise --
        timer.start("deno")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            if temporal_chop is False:
                deno = model(noisy/imax,flows=flows)*imax
            else:
                t = noisy.shape[0]
                tchop = 10
                nchop = (t-1) // tchop + 1
                deno = []
                for ichop in range(nchop):
                    t_start = ichop * tchop
                    t_end = min((ichop+1) * tchop,t)
                    tslice = slice(t_start,t_end)
                    flows_t = slice_flows(flows,t_start,t_end)
                    noisy_t = noisy[tslice]
                    deno_t = model(noisy_t/imax,flows=flows_t)*imax
                    deno.append(deno_t)
                deno = th.cat(deno)
        timer.stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)
        deno = deno.clamp(0.,imax)
        flops = 0.

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        # deno_fns = colanet.utils.io.save_burst(deno,out_dir,"deno")
        deno_fns = []
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
        results.flops.append([flops])
        for name,time in timer.items():
            results[name].append(time)

    return results

def load_trained_state(model,use_train,ca_fwd,sigma,ws,wt):

    if ca_fwd == "stnls_k":
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
    model.model.load_state_dict(state)
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
    # cache_name = "explore_grid" # current!
    cache_name = "explore_grid_0"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    # cfg.isize = "256_256"
    # cfg.isize = "none"#"128_128"
    cfg.bw = True
    cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1

    # -- get mesh --
    dnames,sigmas = ["set8"],[10]#,30.]
    # vid_names = ["tractor"]
    vid_names = ["sunflower"]
    # vid_names = ["sunflower","hypersmooth","tractor"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]
    # ws,wt,k,sb = [10,15,20,25,30],[0,1,2,3,5],[100],[256,1024,10*1024]#1024*1]
    ws,wt,k,sb = [20],[3],[100],[256,1024,10*1024,128*128*10]#1024*1]
    flow,isizes,adapt_mtypes = ["true"],["128_128"],["rand"]
    ca_fwd_list,use_train = ["stnls_k"],["true"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"ca_fwd":ca_fwd_list,
                 "ws":ws,"wt":wt,"k":k, "sb":sb, "use_chop":["false"]}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg)


    # -- exps [2/3] --
    exp_lists['ws'] = [5]
    exp_lists['wt'] = [2,3,5]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg)

    # -- exps [3/3] --
    flow = ["true"]
    isizes = ["128_128",]
    exp_lists['ws'] = [10,15,20,25,30]
    exp_lists['wt'] = [5,0,1,2,3,4]
    exp_lists['sb'] = [28*1024]
    exps_c = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_c,cfg)

    # -- exps [3/3] --
    exp_lists['ws'] = [20]
    exp_lists['wt'] = [3]
    exp_lists['k'] = [100]
    exp_lists['isize'] = ["512_512","384_384","256_256","128_128","64_64"]
    exp_lists['sb'] = [512*512*3,384*384*3,256*256*3,128*128*3,64*64*3]
    # exp_lists['sb'] = [220*220*3,180*180*3,140*140*3,100*100*3,60*60*3]
    # exp_lists['isize'] = ["220_220","180_180","140_140","100_100","60_60"]
    exps_d = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.nframes = 3
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cache_io.append_configs(exps_d,cfg)

    # -- combine --
    # exps = exps_c
    # exps = exps_d +exps_c #+ exps_d# + exps_b + exps_c
    # cache_io.append_configs(exps,cfg) # merge the two
    # pp.pprint(exps[0])
    # exit(0)
    exps = explore_configs.search_space_cfg()

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):
        # break

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        # if exp.use_chop == "false" and exp.ca_fwd != "stnls_k":
        #     cache.clear_exp(uuid)
        # if exp.ca_fwd != "stnls_k" and exp.sigma == 30.:
        #     cache.clear_exp(uuid)
        # if exp.sigma == 30. and exp.ca_fwd == "stnls_k":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    # records = records[records['sb'].isin([256,1024,10*1024])]
    # print(records.filter(like="timer"))
    # print(records[['ws','wt','sb','mem_res','timer_deno']])

    exit(0)
    # -- viz report --
    for use_train,tdf in records.groupby("use_train"):
        for ca_group,gdf in tdf.groupby("ca_fwd"):
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
