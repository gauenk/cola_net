"""

Non-Local Search Approximates Attenion

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
import svnlb

# -- caching results --
import cache_io

# -- matplotlib --
import matplotlib.pyplot as plt
# improt ma

# -- network --
import colanet
import colanet.configs as configs
from colanet import lightning
from colanet.utils.misc import optional,slice_flows
import colanet.utils.gpu_mem as gpu_mem
from colanet.utils.misc import rslice,write_pickle,read_pickle

def model_from_cfg(cfg):
    # -- network --
    nchnls = 1 if cfg.bw else 3
    model = colanet.refactored.load_model(cfg.mtype,cfg.sigma,2,nchnls).to(cfg.device)
    model.eval()
    model.model.body[8].ca_forward_type = cfg.ca_fwd
    model.model.body[8].ws = cfg.ws
    model.model.body[8].wt = cfg.wt
    model.model.body[8].k = cfg.k
    model.model.body[8].sb = cfg.sb
    use_chop = False
    model.chop = use_chop

    # -- optional load trained weights --
    load_trained_state(model,cfg.use_train,cfg.ca_fwd,cfg.sigma,cfg.ws,cfg.wt)

    return model

def orignial_model_from_cfg(cfg):
    # -- original exec --
    model = colanet.original.load_model(cfg.mtype,cfg.sigma).eval()
    model.chop = False
    model.eval()
    return model

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.errors_m = []
    results.errors_s = []
    # results.psnrs = []
    # results.ssims = []
    # results.noisy_psnrs = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_deno = []
    results.timer_flow = []
    results.timer_deno_og = []
    results.mem_res = []
    results.mem_alloc = []

    # -- get model --
    model = model_from_cfg(cfg)
    model_og = orignial_model_from_cfg(cfg)
    imax = 255.

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
    # -- iterate over indices --
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data.te[index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums']
        print("[%d] noisy.shape: " % index,noisy.shape)

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
            noisy_np = noisy.cpu().numpy()
            if noisy_np.shape[1] == 1:
                noisy_np = np.repeat(noisy_np,3,axis=1)
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- denoise [original] --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        timer.start("deno_og")
        with th.no_grad():
            if cfg.run_ca_fwd == "true":
                deno_og = model_og.ca_forward(noisy/imax)*imax
            else:
                deno_og = model_og(noisy/imax,0,False)*imax
        timer.stop("deno_og")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)
        deno_og = deno_og.clamp(0.,imax)/imax

        # -- denoise [proposed] --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        timer.start("deno")
        with th.no_grad():
            if cfg.run_ca_fwd == "true":
                deno = model.ca_forward(noisy/imax)*imax
            else:
                deno = model(noisy/imax,flows=flows)*imax
        timer.stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)
        deno = deno.clamp(0.,imax)/imax

        # -- compute error --
        errors = th.abs(deno_og - deno)/(deno_og.abs()+1e-5)
        errors_m = th.mean(errors).item()
        errors_s = th.std(errors).item()
        results.errors_m.append(errors_m)
        results.errors_s.append(errors_s)

        # -- save example --
        # out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        # deno_fns = colanet.utils.io.save_burst(deno,out_dir,"deno")
        # colanet.utils.io.save_burst(clean,out_dir,"clean")

        # -- psnr --
        # noisy_psnrs = colanet.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        # psnrs = colanet.utils.metrics.compute_psnrs(deno,clean,div=imax)
        # ssims = colanet.utils.metrics.compute_ssims(deno,clean,div=imax)
        # print(noisy_psnrs)
        # print(psnrs)

        # -- append results --
        # results.psnrs.append(psnrs)
        # results.ssims.append(ssims)
        # results.noisy_psnrs.append(noisy_psnrs)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results

def load_trained_state(model,use_train,ca_fwd,sigma,ws,wt):

    # -- skip if needed --
    if not(use_train == "true"): return

    # -- model path --
    if ca_fwd == "stnls_k":
        if sigma == 50.:
            model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=38-val_loss=1.15e-03.ckpt"
        else:#sigma == 30:
            model_path = "output/checkpoints/aa543914-3948-426b-b744-8403d46878cd-epoch=30.ckpt"
    elif ca_fwd == "default":
        model_path = "output/checkpoints/dec78611-36a7-4a9e-8420-4e60fe8ea358-epoch=91-val_loss=6.63e-04.ckpt"
    else:
        raise ValueError(f"Uknown ca_fwd [{ca_fwd}]")

    # -- load model state --
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    model.model.load_state_dict(state)
    return model


def run_plot(records):
    # -- filter --
    records = records[records['ca_fwd'] == "stnls_k"]

    # -- log --
    records = records.reset_index(drop=True)
    idx = records.index
    data = records.loc[:,'errors_m'].astype(np.float32)
    records.loc[idx,'errors_m'] = np.log10(data)
    data = records.loc[:,'errors_s'].astype(np.float32)
    # records.loc[idx,'errors_s'] = np.log10(data)


    # -- plot constants --
    FSIZE = 18
    FSIZE_B = 20
    FSIZE_S = 15
    SAVE_DIR = Path("output/plots/")

    # -- init plot --
    ginfo = {'width_ratios': [0.47],'wspace':0.05, 'hspace':0.0,
             "top":0.88,"bottom":0.22,"left":0.115,"right":0.99}
    fig,ax = plt.subplots(1,1,figsize=(8,3),gridspec_kw=ginfo)

    # -- colors to nbwd --
    lines = ['-','--']
    colors = ["blue","orange"]
    Z = np.sqrt(3*128*128)

    # -- two types --
    b = 0
    for run_ca_fwd,cdf in records.groupby("run_ca_fwd"):

        # -- unpack --
        yvals = np.stack(cdf['errors_m'].to_numpy()).ravel()
        # yerr = np.stack(cdf['errors_s'].to_numpy()).ravel()/Z
        xvals = np.stack(cdf['k'].to_numpy()).ravel()

        # -- plot --
        label = "Attn." if run_ca_fwd == "true" else "Final"
        color = colors[b]
        # ax.errorbar(xvals, yvals, yerr=yerr,color=color, label=label,
        #             linewidth=3)
        ax.plot(xvals, yvals, color=color, label=label,linewidth=3,
                marker='x',markersize=10)
        b+=1

    # -- compute ticks --
    y = records['errors_m']
    x = records['k'].to_numpy()
    x = np.sort(np.unique(x))
    xmin,xmax = x.min().item(),x.max().item()
    ymin,ymax = y.min().item()*1.1,y.max().item()*1.1
    print(ymin,ymax)
    yticks = np.linspace(ymin,ymax,5)
    print(ymin,ymax)
    yticklabels = ["%1.1f" % x for x in yticks]
    xticks = np.linspace(xmin,xmax,4)
    xticklabels = ["%d" % x for x in xticks]

    # -- set ticks --
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE)
    ax.set_ylim(ymin,ymax)

    # -- set labels --
    ax.set_ylabel("Log10 Relative Error",fontsize=FSIZE)
    ax.set_xlabel("Num. of Neighbors",fontsize=FSIZE)
    # ax.axhline(np.log10(1e-5),color='k',linestyle='--')

    # -- format titles --
    ax.set_title("Non-Local Search to Approximate Attention",fontsize=FSIZE_B)

    # -- legend --
    # 0.65,1.08
    leg1 = ax.legend(bbox_to_anchor=(-0.01,0.5), loc="upper left",fontsize=FSIZE,
                     title="Comparison",title_fontsize=FSIZE,framealpha=1.,
                     edgecolor='k',ncol=2)
    leg1.get_frame().set_alpha(None)
    leg1.get_frame().set_facecolor((0, 0, 0, 0.0))

    # -- save figure --
    root = SAVE_DIR
    if not root.exists(): root.mkdir(parents=True)
    fn = root / "nls_approx_attn.png"
    plt.savefig(str(fn),dpi=800,transparent=True)


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "nls_approx_attn" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    # cfg.isize = "256_256"
    # cfg.isize = "none"#"128_128"
    cfg.bw = True
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1

    # -- get mesh --
    dnames,sigmas = ["set8"],[50]#,30.]
    vid_names = ["tractor"]
    # vid_names = ["sunflower"]
    # vid_names = ["sunflower","hypersmooth","tractor"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]
    ws,wt,sb = [-1],[0],[48*1024]#1024*1]
    k = [100,500,1000,2000,4000,8000,12000,13000,14000,15000,16384]
    # k = [100,16384]
    flow,isizes,adapt_mtypes = ["false"],["128_128"],["rand"]
    ca_fwd_list,use_train = ["stnls_k"],["false"]
    run_ca_fwd = ["true","false"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"ca_fwd":ca_fwd_list,
                 "ws":ws,"wt":wt,"k":k, "sb":sb, "use_chop":["false"],
                 "run_ca_fwd":run_ca_fwd}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two

    # -- original w/out training --
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exp_lists['ca_fwd'] = ["default"]
    exp_lists['use_chop'] = ["false"]
    exp_lists['sb'] = [1]
    exp_lists['k'] = [-1]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.bw = True
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a + exps_b

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
        # if exp.k == -1:
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records[['errors_m','k','run_ca_fwd','ca_fwd']])
    run_plot(records)

if __name__ == "__main__":
    main()

