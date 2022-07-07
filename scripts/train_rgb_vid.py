

# -- misc --
import os,math,tqdm
import pprint,copy
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

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only


class ColaNetLit(pl.LightningModule):

    def __init__(self,mtype,sigma,batch_size=1,flow=True,
                 ensemble=False,ca_fwd="dnls_k",isize=None):
        super().__init__()
        self.mtype = mtype
        self.sigma = sigma
        self._model = [colanet.refactored.load_model(mtype,sigma)]
        self.net = self._model[0].model
        self.net.body[8].ca_forward_type = ca_fwd
        self.batch_size = batch_size
        self.flow = flow
        self.isize = isize

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
            noisy_np = vid.cpu().numpy()
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

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),lr=5e-4)
        return optim

    def training_step(self, batch, batch_idx):

        # -- get data --
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        region = batch['region'][0]
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- foward --
        deno = self.forward(noisy)

        # -- report loss --
        loss = th.mean((clean - deno)**2)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False,
                 batch_size=self.batch_size)

        # -- update --
        return loss

    def validation_step(self, batch, batch_idx):

        # -- denoise --
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        region = batch['region'][0]
        # print(region)
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        deno = self.forward(noisy)
        mem_gb = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1)
        self.log("val_gpu_mem", mem_gb, on_step=False,
                 on_epoch=True,batch_size=1)

    def test_step(self, batch, batch_nb):

        # -- denoise --
        index = batch['index'][0]
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        region = batch['region'][0]
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        deno = self.forward(noisy)
        mem_gb = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_gpu_mem = mem_gb
        results.test_index = index.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print(metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)


def launch_training(cfg):

    # -=-=-=-=-=-=-=-=-
    #
    #     Training
    #
    # -=-=-=-=-=-=-=-=-

    # -- network --
    model = ColaNetLit(cfg.mtype,cfg.sigma,cfg.batch_size,
                       cfg.flow=="true",cfg.ensemble=="true",
                       cfg.ca_fwd,cfg.isize)

    # -- create timer --
    timer = ExpTimer()

    # -- load dataset with testing mods isizes --
    model.isize = None
    cfg_clone = copy.deepcopy(cfg)
    cfg_clone.isize = None
    cfg_clone.nsamples = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- validation performance --
    timer.start("init_val_te")
    init_val_report = MetricsCallback()
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,callbacks=[init_val_report])
    trainer.test(model, loaders.val)
    init_val_results = val_report.metrics
    init_val_res_fn = save_dir / "init_val.pkl"
    write_pickle(init_val_res_fn,init_val_results)
    timer.stop("init_val_te")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Training
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reset model --
    model.isize = cfg.isize

    # -- data --
    data,loaders = data_hub.sets.load(cfg)

    # -- pytorch_lightning training --
    chkpt_fn = cfg.uuid + "-{epoch:02d}-{val_loss:.2f}"
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=3,mode="max",
                                          dirpath=cfg.checkpoint_dir,filename=chkpt_fn)
    timer.start("train")
    trainer = pl.Trainer(gpus=2,precision=32,limit_train_batches=1.,
                         max_epochs=cfg.nepochs,log_every_n_steps=1,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, loaders.tr, loaders.val)
    timer.stop("train")
    best_path = checkpoint_callback.best_model_path

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Validation Testing
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reload dataset with no isizes --
    model.isize = None
    cfg_clone = copy.deepcopy(cfg)
    cfg_clone.isize = None
    cfg_clone.nsamples = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- prepare save directory --
    save_dir = Path("./output/training/") / cfg.uuid
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # -- training performance --
    timer.start("train_te")
    tr_report = MetricsCallback()
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,callbacks=[tr_report])
    trainer.test(model, loaders.tr)
    tr_results = tr_report.metrics
    tr_res_fn = save_dir / "train.pkl"
    write_pickle(tr_res_fn,tr_results)
    timer.stop("train_te")

    # -- validation performance --
    timer.start("val_te")
    val_report = MetricsCallback()
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,callbacks=[val_report])
    trainer.test(model, loaders.val)
    val_results = val_report.metrics
    val_res_fn = save_dir / "val.pkl"
    write_pickle(val_res_fn,val_results)
    timer.stop("val_te")

    # -- report --
    results = edict()
    results.best_path = best_path
    results.train_results_fn = tr_res_fn
    results.init_val_results_fn = init_val_res_fn
    results.val_results_fn = val_res_fn
    results.train_time = timer["train"]
    results.test_train_time = timer["train_te"]
    results.test_val_time = timer["val_te"]
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
    return cfg

def test_tuned_models(cfg):
    pass

def main():

    # -- print os pid --
    print("PID: ",os.getpid())

    # -- init --
    verbose = True
    cache_dir = ".cache_io"
    cache_name = "train_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- create exp list --
    ws,wt = [10],[5]
    sigmas = [50.]#,30.,10.]
    isizes = ["96_96"]
    # ca_fwd_list = ["default","dnls_k"]
    ca_fwd_list = ["dnls_k","default"]
    exp_lists = {"sigma":sigmas,"ws":ws,"wt":wt,"isize":isizes,
                 "ca_fwd":ca_fwd_list}
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
            results = launch_training(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- results --
    records = cache.load_flat_records(exps)
    print(records.columns)
    print(records['uuid'])
    print(records['checkpoint_dir'].iloc[0])
    print(records['checkpoint_dir'].iloc[1])
    print(records['best_path'].iloc[0])
    print(records['best_path'].iloc[1])

    # -- load res --
    uuids = list(records['uuid'].to_numpy())
    cas = list(records['ca_fwd'].to_numpy())
    fns = list(records['val_results_fn'].to_numpy())
    res_a = read_pickle(fns[0])
    res_b = read_pickle(fns[1])
    print(uuids,cas,fns)
    print(res_a['test_psnr'])
    print(res_b['test_psnr'])


# def find_records(path,uuid):
#     files = []
#     for fn in path.iterdir():
#         if uuid in fn:
#             files.append(fn)
#     for fn in files:
#         fn.



if __name__ == "__main__":
    main()
