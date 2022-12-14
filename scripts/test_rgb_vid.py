
# -- misc --
import os,math,tqdm
import pprint,random,copy
pp = pprint.PrettyPrinter(indent=4)
from functools import partial


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
# from colanet import flow
from dev_basics import flow

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
from colanet.utils.aug_test import test_x8

def run_exp(_cfg):

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

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
    groups = data[cfg.dset].groups
    # indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     cfg.frame_start,cfg.nframes)

    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data[cfg.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        print(th.mean((noisy - clean)**2))
        vid_frames = sample['fnums'].numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)
        temporal_chop = noisy.shape[0] > 20
        temporal_chop = temporal_chop and not(use_chop)
        # noisy,clean = noisy[...,:1,:,:],clean[...,:1,:,:]

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)
        noisy_psnrs = colanet.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        print(noisy_psnrs)

        # -- create timer --
        timer = colanet.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 390*39#ngroups*1024

        # -- optical flow --
        timer.start("flow")
        flows = flow.orun(noisy,cfg.flow)
        # if cfg.flow is True:
        #     sigma_est = flow.est_sigma(noisy)
        #     flows = flow.run_batch(noisy[None,:],sigma_est)
        # else:
        #     flows = flow.run_zeros(noisy[None,:])
        timer.sync_stop("flow")

        # -- denoise --
        if cfg.aug_test:
            aug_fxn = partial(test_x8,model,use_refine=cfg.aug_refine_inds)
        else: aug_fxn = model
        fwd_fxn = get_fwd_fxn(cfg,aug_fxn)

        # -- run once for setup gpu --
        if cfg.burn_in:
            with th.no_grad():
                fwd_fxn(noisy[[0],...,:128,:128]/imax,None)
            model.reset_times()

        # -- benchmark it! --
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
        print(noisy.min(),noisy.max())
        print(clean.min(),clean.max())
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
            if not(name in results):
                results[name] = []
            results[name].append(time)
        for name,time in model.times.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    return results

def load_trained_state(model,use_train,ca_fwd,sigma,ws,wt):

    # -- skip if needed --
    if not(use_train is True): return

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
    cfg.isize = "512_512" # a fixed training parameters
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
    print("Loading trained network: ",model_path)
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

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    # cache_name = "test_rgb_net" # best results (aaai23)
    # cache_name = "test_rgb_net_cvpr23" # best results
    cache_name = "test_rgb_net_icml23" # best results
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    # cfg.isize = "128_128"
    # cfg.isize = "256_256"
    # cfg.isize = "512_512"
    cfg.isize = "none"#"128_128"
    # cfg.isize = "256_256"
    cfg.bw = True
    cfg.n_colors = 1
    # cfg.bw = False
    # cfg.n_colors = 3
    cfg.nframes = 0
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.attn_mode = "dnls_k"
    # cfg.aug_test = True
    # cfg.aug_refine_inds = True
    # cfg.aug_refine_inds = False
    cfg.return_inds = False
    cfg.arch_return_inds = False
    cfg.burn_in = False
    # cfg.flow = True
    cfg.pretrained_load = True
    cfg.pretrained_type = "lit"

    # -- processing --
    cfg.spatial_crop_size = 540
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 5#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.softmax_scale = 10.

    # -- get mesh --
    dnames = ["set8"]
    dset = ["te"]
    sigmas = [50]
    sigmas = [30]
    # sigmas = [25,30]
    # vid_names = ["tractor"]
    # vid_names = ["sunflower","tractor","park_joy"]
    # vid_names = ["sunflower"]
    # vid_names = ["sunflower","hypersmooth","tractor"]
    vid_names = ["snowboard","sunflower","tractor","motorbike",
                 "hypersmooth","park_joy","rafting","touchdown"]

    # -- new mesh --
    # dnames = ["submillilux_real"]
    # vid_names = ["0"]
    dnames = ["davis"]
    dset = ["val"]
    vid_names = ["bike-packing", "blackswan", "bmx-trees", "breakdance",
                 "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
                 "dog", "dogs-jump", "drift-chicane", "drift-straight", "goat",
                 "gold-fish", "horsejump-high", "india", "judo", "kite-surf",
                 "lab-coat", "libby", "loading", "mbike-trick", "motocross-jump",
                 "paragliding-launch", "parkour", "pigs", "scooter-black",
                 "shooting", "soapbox"]

    # -- standard --
    # ws,wt = [27],[3]
    # cfg.k_s = 100
    # cfg.k_a = 100
    # cfg.refine_inds = "f-f-f"

    # -- prop0 --
    # cfg.k_s = 100
    # cfg.k_a = 100
    cfg.k_s = 80
    cfg.k_a = 80
    # cfg.ws = 21
    # cfg.wt = 3
    # cfg.ws_r = '1-1-1'
    # cfg.k_a = 50
    # cfg.refine_inds = "f-f-f"

    # -- prop1 --
    # ws,wt = ['27-27-3'],['3-3-0']
    # cfg.k_s = '100-500-100'
    # cfg.k_a = '100-100-100'
    # cfg.k_a = 50
    # cfg.refine_inds = "f-f-t"

    # -- prop2 --
    # ws = [27]
    ws = [15]#,21,27]
    wt = [0,3]
    # wt = [3,0]
    # ws,wt = ['27-27-27'],['3-3-3']
    # cfg.k_s = '100-100-100'
    # cfg.k_a = '100-100-100'
    # cfg.refine_inds = "f-f-f"

    flow = ["true"]#,"false"]
    ca_fwd_list = ["dnls_k"]
    use_train = ["false"]
    # refine_inds = ["f-f-f","f-f-t","f-t-f","f-t-t"]
    # aug_test = ["false"]
    refine_inds = ["f-f-f"]
    # refine_inds = ["f-f-f","t-t-t","f-f-t"]
    aug_test = ["false","true"]
    # aug_test = ["false","true"]
    aug_refine_inds = ["false"]
    # aug_refine_inds = ["true"]

    # refine_inds = ["f-f-f","t-t-t","f-f-t"]
    # refine_inds = ["f-f-f","t-t-t"]
    # aug_test = ["false","true"]
    # aug_refine_inds = ["true"]
    # ws_r = [1]

    # aug_test = ["false","true"]
    # aug_refine_inds = ["false","true"]
    # ws_r = [1,3]
    model_type = ['augmented']
    pretrained_path = [
        "2539a251-8233-49a8-bb4f-db68e8c96559-epoch=38-val_loss=1.15e-03.ckpt" # FT-50
        # "aa543914-3948-426b-b744-8403d46878cd-epoch=30.ckpt", # FT-30
        # "81e83985-53b9-47ef-b201-1cbcd76cc20a-epoch=19.ckpt",
        # "f542a93c-edd0-458c-8b7d-a47430b64adc-epoch=01.ckpt",
        # "81e83985-53b9-47ef-b201-1cbcd76cc20a-epoch=19.ckpt",
        # "f542a93c-edd0-458c-8b7d-a47430b64adc-epoch=01.ckpt",
        # "f542a93c-edd0-458c-8b7d-a47430b64adc-epoch=04.ckpt",
        "f542a93c-edd0-458c-8b7d-a47430b64adc-epoch=08.ckpt",
        # "20c5e28c-3333-484c-b982-bca1bf25eb19-epoch=03.ckpt"
    ]
    # ws_r = [1,3]
    ws_r = [1]
    exp_lists = {"dname":dnames,"dset":dset,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"use_train":use_train,"ca_fwd":ca_fwd_list,
                 "use_chop":["false"],"ws_r":ws_r,"wt":wt,"ws":ws,
                 "model_type":model_type,"refine_inds":refine_inds,
                 "aug_refine_inds":aug_refine_inds,"aug_test":aug_test,
                 "pretrained_path":pretrained_path}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two


    # -- compare with smaller search --
    # exp_lists['vid_names'] = ["sunflower"]
    exp_lists['aug_test'] = ["false","true"]
    exp_lists['refine_inds'] = ["f-f-f"]
    exp_lists['ws_r'] = [-1]
    cfg.ws = '27-9-9'
    cfg.wt = '3-0-0'
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # # -- original w/out training --
    # cfg.ws = 27
    # # exp_lists['model_type'] = ['original']
    # exp_lists['k'] = [100]
    # exp_lists['sb'] = [48*1024]
    # exp_lists['model_type'] = ['refactored']
    # exp_lists['flow'] = ["false"]
    # exp_lists['use_train'] = ["false"]#,"true"]
    # exp_lists['ca_fwd'] = ["default"]
    # exp_lists['use_chop'] = ["false"]
    # exps_c = cache_io.mesh_pydicts(exp_lists) # create mesh
    # cfg.bw = True
    # cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a# + exps_b
    # exps = exps_b

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
        # if "submillilux" in exp.dname:
        #     cache.clear_exp(uuid)
        # cache.clear_exp(uuid)
        cache.clear_exp(uuid)
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

    # -- neat report --
    fields = ['pretrained_path','refine_inds','use_chop','aug_test','sigma','vid_name','ws','ws_r','wt']
    fields_summ = ['pretrained_path','refine_inds','use_chop','aug_test','ws','ws_r','wt']
    res_fields = ['psnrs','ssims','timer_deno','mem_alloc','mem_res']
    res_fmt = ['%2.3f','%1.3f','%2.3f','%2.3f','%2.3f','%2.3f']
    # res_fields = ['psnrs','ssims','timer_deno','timer_extract',
    #               'timer_search','timer_agg','mem_alloc','mem_res']
    # res_fmt = ['%2.3f','%1.3f','%2.3f','%2.3f','%2.3f','%2.3f','%2.3f','%2.3f']

    # -- run agg --
    agg = {key:[np.stack] for key in res_fields}
    grouped = records.groupby(fields).agg(agg)
    grouped.columns = res_fields
    grouped = grouped.reset_index()
    for field in res_fields:
        grouped[field] = grouped[field].apply(np.mean)
    res_fmt = {k:v for k,v in zip(res_fields,res_fmt)}
    print("\n\n" + "-="*10+"-" + "Report" + "-="*10+"-")
    for sigma,sdf in grouped.groupby("sigma"):
        print("--> sigma: %d <--" % sigma)
        for gfields,gdf in sdf.groupby(fields_summ):
            gfields = list(gfields)

            # -- header --
            header = "-"*5 + " ("
            for i,field in enumerate(fields_summ):
                if field == "pretrained_path":
                    path_str = get_pretrained_path_str(gfields[i])
                    header += "%s, " % (path_str)
                else:
                    header += "%s, " % (gfields[i])
            header = header[:-2]
            header += ") " + "-"*5
            print(header)

            for vid_name,vdf in gdf.groupby("vid_name"):
                # -- res --
                res = "%13s: " % vid_name
                for field in res_fields:
                    res += res_fmt[field] % (vdf[field].mean()) + " "
                print(res)
            # -- res --
            res = "%13s: " % "Ave"
            for field in res_fields:
                res += res_fmt[field] % (gdf[field].mean()) + " "
            print(res)

def get_pretrained_path_str(path):
    path = str(Path(path).stem)
    if "epoch" in path:
        epoch = path.split("=")[1].split(".")[0]
        uuid = path.split("-")[0][:4]
        pstr = "%s-%s" % (uuid,epoch)
    elif path == "none":
        pstr = "none"
    else:
        pstr = path
    return pstr

if __name__ == "__main__":
    main()
