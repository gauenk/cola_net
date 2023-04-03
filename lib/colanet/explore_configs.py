from . import configs
import cache_io


def search_space_cfg():

    #
    # -- baseline --
    #
    cfg = configs.default_test_vid_cfg()
    cfg.bw = True
    cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1

    #
    # -- our config --
    #

    dnames,sigmas = ["set8"],[10]#,30.]
    vid_names = ["sunflower"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    # ws,wt,k,sb = [10,15,20,25,30],[0,1,2,3,5],[100],[256,1024,10*1024]#1024*1]
    sb = [10*128*128]
    wt = [5,0,1,2,3,4]
    ws = [10,15,20,25,30]
    isize = ["128_128"]
    k = [100]
    flow,isizes,adapt_mtypes = ["true"],["128_128"],["rand"]
    ca_fwd_list,use_train = ["stnls_k"],["true"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"ca_fwd":ca_fwd_list,
                 "k":k, "sb":sb, "use_chop":["false"]}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps,cfg)

    return exps
