import dnls
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from einops import rearrange,repeat
from colanet.utils.misc import assert_nonan
import colanet
from colanet.utils.misc import optional
from torch.nn.functional import unfold as th_unfold

"""
fundamental functions
"""
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches_og(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same', region=None):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    images_pad, paddings = same_padding(images, ksizes, strides, rates)
    # if padding == 'same':
    #     images, paddings = same_padding(images, ksizes, strides, rates)
    # elif padding == 'valid':
    #     pass
    # else:
    #     raise NotImplementedError('Unsupported padding type: {}.\
    #             Only "same" or "valid" are supported.'.format(padding))
    # print("images.shape: ",images.shape,ksizes,strides)
    t,c,h,w = images.shape
    ksize = ksizes[0]
    adj = (ksize//2) - paddings[0]
    stride = strides[0]
    coords = [0,0,h,w] if (region is None) else region[2:]
    adj = 0
    unfold = dnls.iUnfold(ksize,coords,stride=stride,dilation=1,
                          adj=adj,only_full=False,border="zero")
    patches = unfold(images)
    patches_a = rearrange(patches,'(t n) 1 1 c h w -> t (c h w) n',t=t)
    patches = patches_a

    # unfold = dnls.iunfold.iUnfold(ksize,coords,stride=stride,dilation=1,
    #                               match_nn=True)#adj=ps//2,only_full=True)
    # patches = unfold(images_pad)
    # print(patches[0,0,0,0])
    # patches_b = rearrange(patches,'(t n) 1 1 c h w -> t (c h w) n',t=t)
    # print(patches_b.shape)

    # diff = th.abs(patches_a - patches_b).mean(1)
    # diff = rearrange(diff,'t (h w) -> t 1 h w',h=h//4)
    # dnls.testing.data.save_burst(diff,'output/ca/','diff')
    # error = diff.sum()
    # print("error: ",error)
    # exit(0)

    # print("[1] patches.shape: ",patches.shape)
    # folder = dnls.ifold.iFold((T,C,H,W),coords,stride=stride,dilation=1,adj=True)
    # unfold = torch.nn.Unfold(kernel_size=ksizes,padding=0,stride=strides)
    # patches = unfold(images)
    # print("[2] patches.shape: ",patches.shape)

    return patches, paddings

"""
CA network
"""
class ContextualAttention_Enhance(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
                 , inter_channels=16,use_multiple_size=False,use_topk=False,add_SE=False):
        super(ContextualAttention_Enhance, self).__init__()
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.use_topk=use_topk
        self.add_SE=add_SE
        # self.SE=SE_net(in_channels=in_channels)
        self.conv33=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def dnls_k_forward(self, b, region=None, flows=None, exact=False, ws=29, wt=0, k=100,sb=None):

        # -- get images --
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = self.phi(b)
        region = None
        reflect_bounds = False

        # -- unpack parameters --
        t,c,h,w = b1.shape
        kernel = self.ksize
        vshape = b1.shape
        ps = self.ksize
        stride0 = self.stride_1 # 4
        stride1 = self.stride_2 # 1
        chnls = b2.shape[1]
        dil,adj,pt = 1,0,1
        use_search_abs = ws == -1
        region = [0,0,h,w] if region is None else region
        device = b.device

        # -- global region --
        use_k = not(ws==-1)
        use_search_abs = ws == -1

        # -- local region --
        # ws,wt = 40,0
        # k = 250
        # use_k = True
        # use_search_abs = False

        # -- get search size --
        cr_h = region[2] - region[0]
        cr_w = region[3] - region[1]

        # -- batching params --
        nh = (cr_h-1)//stride0+1
        nw = (cr_w-1)//stride0+1
        npix = h * w
        ntotal = t * nh * nw
        # bdiv = t if self.training else 1
        # nbatch = ntotal//(t*bdiv)
        # nbatch = min(nbatch,4096)
        # nbatch = min(nbatch,4096*2)
        if sb is None:
            div = 2 if npix >= (540 * 960) else 1
            nbatch = ntotal//(t*div)
        else:
            nbatch = sb
        # nbatch = nbatch if use_k else min(nbatch,2048)
        nbatches = (ntotal-1) // nbatch + 1
        # ws,wt = -1,0

        # -- offsets --
        oh0,ow0,oh1,ow1 = 1,1,3,3

        # -- define functions --
        ifold = dnls.iFoldz(vshape,region,stride=stride0,dilation=dil,
                                 adj=0,only_full=False,use_reflect=False,
                                 device=device)
        # wfold = dnls.iFold(vshape,region,stride=stride0,dilation=dil,
        #                          adj=0,only_full=False,use_reflect=False,
        #                          device=device)
        # scatter = dnls.scatter.ScatterNl(ps,pt,exact=exact,adj=0,reflect_bounds=False)
        iunfold = dnls.iUnfold(ps,region,stride=stride1,dilation=dil,
                               adj=0,only_full=False,border="zero")
        fflow = optional(flows,'fflow',None)
        bflow = optional(flows,'bflow',None)
        # print(ws,wt,k)
        xsearch = dnls.search.init("prod_with_index",fflow, bflow, k, ps, pt, ws, wt,
                                   oh0, ow0, oh1, ow1, chnls=-1,
                                   dilation=dil, stride0=stride0,stride1=stride1,
                                   reflect_bounds=False, use_k=use_k,use_adj=True,
                                   use_search_abs=use_search_abs, exact=exact)
        wpsum = dnls.reducers.WeightedPatchSum(ps, pt, h_off=0, w_off=0, dilation=dil,
                                               reflect_bounds=reflect_bounds,
                                               adj=0, exact=exact)

        # -- misc --
        # raw_int_bs = list(b1.size())  # b*c*h*w

        # patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
        #                                               strides=[self.stride_1, self.stride_1],
        #                                               rates=[1, 1],
        #                                               padding='same',region=region)
        # # print("patch_28.shape: ",patch_28.shape)
        # patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        # patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        # patch_28_group = torch.split(patch_28, 1, dim=0)

        # patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
        #                                                 strides=[self.stride_2, self.stride_2],
        #                                                 rates=[1, 1],padding='same',region=None)
        # print("patch_112.shape: ",patch_112.shape)
        # patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        # patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        # patch_112_group = torch.split(patch_112, 1, dim=0)

        # patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
        #                                                 strides=[self.stride_2, self.stride_2],
        #                                                 rates=[1, 1],
        #                                                 padding='same',region=None)
        # patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        # patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        # patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)

        # -- batch across queries --
        for index in range(nbatches):

            # -- timer --
            # print("%d/%d" % (index+1,nbatches))
            timer = colanet.utils.timer.ExpTimer()

            # -- batch info --
            qindex = min(nbatch * index,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- get patches --
            # iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch_i,stride0,
            #                                             region,t,device=device)
            # th.cuda.synchronize()

            # -- search --
            # print(iqueries)
            timer.start("xsearch")
            # dists,inds = xsearch(b1,iqueries,b3)
            dists,inds = xsearch(b1,qindex,nbatch_i,b3)
            # print("ref: ",dists[:3,:3])
            # print("dists.shape: ",dists.shape)

            # patch_112_group_2

            # th.cuda.synchronize()
            timer.stop("xsearch")
            # nlDists_nk,nlInds_nk = xsearch_nk(b1,iqueries,b3)
            # if nlDists_nk.ndim == 3:
            #     nlDists_nk = rearrange(nlDists_nk,'d0 h w -> d0 (h w)')
            # print(dists[:3,:3])

            # -- attn mask --
            timer.start("misc")
            yi = F.softmax(dists*self.softmax_scale,1)
            assert_nonan(yi)
            # yi_nk = F.softmax(nlDists_nk*self.softmax_scale,1)
            # assert_nonan(yi_nk)

            # -- get top k patches --
            # yi = yi.detach()
            zi = wpsum(b2,yi,inds).view(nbatch_i,-1)
            # print(zi.shape)

            #
            # -- passes --
            #

            # print("patch_112_group[index].shape: ",patch_112_group[index].shape)
            # pi = patch_112_group[index].view(h*w,-1)
            # zi = torch.mm(yi, pi)

            # print("zi.shape: ",pi.shape)
            # print(zi[:3,:3])
            # print(zi_p[:3,:3])
            # error = th.abs(zi - zi_p).mean()
            # print(error)
            # error = th.abs(zi - zi_p).max()
            # print(error)
            # diff = th.abs(zi - zi_p)
            # args = th.where(diff>1e-2)
            # print(zi[args][:5])
            # print(zi_p[args][:5])
            # exit(0)

            # if use_k:
            #     zi = wpsum(b2,yi,inds).view(iqueries.shape[0],-1)
            #     # yi = yi[...,None].type(th.float64)
            #     # patches_i = scatter(b2,inds).type(th.float64)
            #     # patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
            #     # assert_nonan(patches_i)
            #     # _,k,dim = patches_i.shape
            #     # zi = th.sum(yi * patches_i,1).type(th.float32)
            # else:
            #     # -- scatter new vid type for each ti --
            #     # print(yi.shape,patches[index].shape)
            #     # zi = th.matmul(yi, patches[index])
            #     zi = []
            #     for ti in range(t):
            #         args_i = th.where(iqueries[:,0] == ti)
            #         zi_i = th.mm(yi[args_i], patches[ti])
            #         zi.append(zi_i)
            #     zi = th.cat(zi)
            #     assert_nonan(zi)
            # zi_dnls = zi
            # th.cuda.synchronize()
            timer.stop("misc")

            # -- compare patches --
            # print(patches[index][0].view(-1,ps,ps)[0])
            # print(patches_i[0].view(-1,ps,ps)[0])
            # error = th.abs(patches[index] - patches_i).sum().item()
            # print("Error: ",error)

            # -- testing patches --
            # zi_check = []
            # print(yi_nk.shape)
            # for ti in range(t):
            #     args_i = th.where(iqueries[:,0] == ti)
            #     zi_i = th.matmul(yi_nk[args_i], patches[ti])
            #     zi_check.append(zi_i.type(th.float32))
            # zi_check = th.cat(zi_check)
            # assert_nonan(zi_check)
            # zi = zi_check

            # -- testing --
            # print(zi_dnls[:3,:3])
            # print(zi_check[:3,:3])
            # print(zi_dnls[10:12,10:12])
            # print(zi_check[10:12,10:12])
            # error = th.sum(th.abs(zi_check - zi_dnls)).item()
            # print("Error: ",error)
            # exit(0)

            # -- ifold --
            timer.start("fold")
            # print("zi.shape: ",zi.shape)
            _zi = rearrange(zi,'n (c h w) -> n 1 1 c h w',h=ps,w=ps)
            # ones = th.ones_like(_zi)
            ifold(_zi,qindex)
            # wfold(ones,qindex)
            # th.cuda.synchronize()
            timer.stop("fold")
            # print(timer)

        # -- get post-attn vid --
        y,Z = ifold.vid,ifold.zvid
        # y = ifold.vid
        # Z = wfold.vid
        # y = th.cat(agg)
        # Z = th.cat(wagg)
        # print("[final] y.shape: ",y.shape)
        assert_nonan(y)
        # y_s = y/y.max()
        # dnls.testing.data.save_burst(y_s[:,:3],"./output/ca","y")
        # assert_nonan(Z)
        # Z_s = Z/Z.max()
        # dnls.testing.data.save_burst(Z_s[:,:3],"./output/ca/","z")

        y = y / Z
        # assert_nonan(y)
        # yz_s = y/y.max()
        # dnls.testing.data.save_burst(yz_s[:,:3],"./output/ca/","yz")

        # -- final transform --
        y = self.W(y)
        y = b + y

        if self.add_SE:
            y_SE=self.SE(y)
            y=self.conv33(torch.cat((y_SE*y,y),dim=1))

        return y

    def dnls_forward(self, b, region=None, flows=None, exact=False):

        # -- get images --
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = self.phi(b)

        # -- unpack parameters --
        t,c,h,w = b1.shape
        kernel = self.ksize
        vshape = b1.shape
        ps = self.ksize
        stride0 = 4#self.stride_1
        stride1 = 1#self.stride_2
        chnls = b2.shape[1]
        dil,adj = 1,0
        ws,pt,wt = -1,1,0
        region = [0,0,h,w] if region is None else region
        device = b.device

        # -- get search size --
        cr_h = region[2] - region[0]
        cr_w = region[3] - region[1]

        # -- batching params --
        nh = (cr_h-1)//stride0+1
        nw = (cr_w-1)//stride0+1
        ntotal = t * nh * nw
        nbatch = ntotal//t
        nbatches = (ntotal-1) // nbatch + 1
        # print(nbatch)

        # -- offsets --
        # oh0,ow0,oh1,ow1 = 3,3,1,1
        oh0,ow0,oh1,ow1 = 1,1,3,3

        # -- define functions --
        ifold = dnls.iFoldz(vshape,region,stride=stride0,dilation=dil,
                                 adj=0,only_full=False,use_reflect=False)
        # wfold = dnls.iFold(vshape,region,stride=stride0,dilation=dil,
        #                          adj=0,only_full=False,use_reflect=False)
        iunfold = dnls.iUnfold(ps,region,stride=stride1,dilation=dil,
                               adj=adj,only_full=False,border="zero")
        # iunfold = dnls.iunfold.iUnfold(ps,region,stride=stride1,dilation=dil,adj=True)
        fflow = optional(flows,'fflow',None)
        bflow = optional(flows,'bflow',None)
        xsearch = dnls.xsearch.CrossSearchNl(fflow, bflow, -1, ps, pt, ws, wt,
                                             oh0, ow0, oh1, ow1,
                                             chnls=-1,dilation=dil,stride=stride1,
                                             reflect_bounds=False,use_k=False,
                                             use_search_abs=True,exact=exact)
        # -- unfold patches --
        # patches = th.nn.functional.unfold(b2,(ps,ps))#,0,-1)
        # b1_ones = th.ones_like(b1)
        # b1 = th.ones_like(b1)
        # patches_a,_ = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
        #                                     strides=[self.stride_1, self.stride_1],
        #                                     rates=[1, 1],padding='same')
        # patches_b,_ = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
        #                                     strides=[self.stride_2, self.stride_2],
        #                                     rates=[1, 1],padding='same')

        # patches,_ = extract_image_patches(b2,ksizes=[self.ksize, self.ksize],
        #                                   strides=[self.stride_2, self.stride_2],
        #                                   rates=[1, 1],padding='same')
        # patches = patches.transpose(2,1)
        # print("patches.shape: ",patches.shape)

        # -- iunfold patches --
        patches = iunfold(b2,0,-1)
        patches = rearrange(patches,'(t n) 1 1 c h w -> t n (c h w)',t=t)
        assert_nonan(patches)
        _,xsize,dim = patches.shape

        # -- batch across queries --
        agg,wagg = [],[]
        for index in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * index,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- get patches --
            iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch_i,stride0,
                                                        region,t,device)
            th.cuda.synchronize()

            # -- search --
            # print(iqueries)
            dists,inds = xsearch(b1,iqueries,b3)
            # dists,inds = xsearch(b1_ones,iqueries,b1)
            # th.cuda.synchronize()
            # print("dists.shape: ",dists.shape)
            # dists = rearrange(dists,'d0 h w -> d0 (h w)')

            # print("patches_a.shape: ",patches_a.shape)
            # print("patches_b.shape: ",patches_b.shape)
            # patches_a = th.ones_like(patches_a)
            # patches_b = th.ones_like(patches_b)
            # score_map = th.matmul(patches_a[index].T,patches_b[index])
            # print("dists.shape: ",dists.shape)
            # exit(0)

            # -- viz --
            # print(dists[:3,:3])
            # print(score_map[:3,:3])

            # print(dists[96:99,96:99])
            # print(score_map[96:99,96:99])


            # -- test --
            # diff = th.abs(dists - score_map)/(th.abs(score_map)+1e-10)
            # error = diff.sum()
            # print(error)
            # error = diff.max()
            # print(error)

            # -- attn mask --
            yi = F.softmax(dists*self.softmax_scale,1)
            assert_nonan(yi)

            # -- scatter new vid type for each ti --
            # zi = th.matmul(yi, patches[index])
            # zi_iu = th.matmul(yi, patches_iu[index])
            # print("patches.shape: ",patches[index].shape)
            # print("patches_iu.shape: ",patches_iu[index].shape)

            # diff = th.abs(zi - zi_iu)/(th.abs(zi)+1e-10)
            # error = diff.sum()
            # print(error)
            # error = diff.max()
            # print(error)
            # exit(0)
            # print("iqueries.shape: ",iqueries.shape)

            #
            # -- passes --
            #
            zi = th.matmul(yi, patches[index])

            #
            # -- misc --
            #

            # print("zi.shape: ",zi.shape)
            # exit(0)
            # zi = []
            # for ti in range(t):
            #     args_i = th.where(iqueries[:,0] == ti)
            #     zi_i = th.matmul(yi[args_i], patches[ti])
            #     zi.append(zi_i)
            # zi = th.cat(zi)
            # assert_nonan(zi)

            # print("zi.shape: ",zi.shape)
            # print("ones.shape: ",ones.shape)

            # -- prepare shape --

            # -- fold into videos --
            # print(qindex)
            # print("zi.shape: ",zi.shape)
            # print("ones.shape: ",ones.shape)
            # ifold(zi,qindex)
            # wfold(ones,qindex)

            # -- ifold --
            # _zi = rearrange(zi,'n (c h w) -> n 1 1 c h w',h=ps,w=ps)
            # ones = th.ones_like(_zi)
            # ifold(_zi,qindex)
            # wfold(ones,qindex)

            # -- fold --
            zi = zi[None,:]
            zi = zi.transpose(2,1)
            ones = th.ones_like(zi)
            zi = th.nn.functional.fold(zi,(h,w),(ps,ps),stride=stride0,padding=3)
            agg.append(zi)
            wi = th.nn.functional.fold(ones,(h,w),(ps,ps),stride=stride0,padding=3)
            wagg.append(wi)

        # -- get post-attn vid --
        y = th.cat(agg)
        Z = th.cat(wagg)
        # y = ifold.vid
        # Z = wfold.vid
        # print("[final] y.shape: ",y.shape)
        # y = ifold.vid
        # Z = wfold.vid
        assert_nonan(y)
        # y_s = y/y.max()
        # dnls.testing.data.save_burst(y_s[:,:3],"./output/ca","y")
        # assert_nonan(Z)
        # Z_s = Z/Z.max()
        # dnls.testing.data.save_burst(Z_s[:,:3],"./output/ca/","z")

        y = y / Z
        # assert_nonan(y)
        # yz_s = y/y.max()
        # dnls.testing.data.save_burst(yz_s[:,:3],"./output/ca/","yz")

        # -- final transform --
        y = self.W(y)
        y = b + y

        if self.add_SE:
            y_SE=self.SE(y)
            y=self.conv33(torch.cat((y_SE*y,y),dim=1))

        return y

        # # -- compute cross-scale search inplace --
        # # fold,wfold = dnls.ifold.iFold(),dnls.ifold.iFold()
        # fold,unfold = th.nn.functional.fold,th.nn.functional.unfold
        # # unfold = dnls.iunfold.iUnfold(ksize,region,stride=stride,dilation=1,adj=True)
        # scatter = dnls.scatter_nl(scale=1)
        # dnls_search = dnls.xsearch.CrossScaleSearch(flows.fflow, flows.bflow, k, ps, pt,
        #                                             ws, wt, chnls=chnls,dilation=1, stride=1)

        # yi = F.softmax(dists*self.softmax_scale,1)
        # patches = scatter_nl(x,queryInds)
        # zi = yi @ patches
        # ones = th.ones_like(zi)
        # zi = fold(zi)
        # ones = wfold(ones)
        # zi = zi / ones
        # y.append(zi)


    def forward(self, b, region=None, flows=None, exact=False):

        kernel = self.ksize
        region = None

        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = self.phi(b)

        raw_int_bs = list(b1.size())  # b*c*h*w
        region = region

        patch_28, paddings_28 = extract_image_patches_og(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],padding='same')
        # print("patch_28.shape: ",patch_28.shape)
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches_og(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],padding='same')
        # print("patch_112.shape: ",patch_112.shape)
        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches_og(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],padding='same')


        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        f_groups = torch.split(b3, 1, dim=0)
        # print("f_groups.shape: ",[f.shape for f in f_groups])
        plist = [patch_112_group_2, patch_28_group, patch_112_group]
        # for p in plist:
        #     print("shape: ",[gr.shape for gr in p])

        y = []
        # -- process each batch separately --
        for xii,xi, wi,pi in zip(f_groups,patch_112_group_2, patch_28_group, patch_112_group):
            # print("xii,xi,wi,pi: ",xii.shape,xi.shape,wi.shape,pi.shape)
            w,h = xii.shape[2], xii.shape[3]
            _, paddings = same_padding(xii, [self.ksize, self.ksize], [1, 1], [1, 1])
            # wi = wi[0]  # [L, C, k, k]
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            # print("[pre] wi.shape: ",wi.shape)
            # print("[pre] xi.shape: ",xi.shape)
            wi = wi.view(wi.shape[0],wi.shape[1],-1)
            xi = xi.permute(0, 2, 3, 4, 1) # keep contiguous?
            xi = xi.view(xi.shape[0],-1,xi.shape[4])
            # print("wi.shape: ",wi.shape)
            # print("xi.shape: ",xi.shape)

            # -- compute cross-scale --
            score_map = torch.matmul(wi,xi) # q * v^T
            # print("score_map.shape: ",score_map.shape)
            score_map = score_map.view(score_map.shape[0],score_map.shape[1],w,h)
            b_s, l_s, h_s, w_s = score_map.shape
            # print("score_map.shape: ",score_map.shape)

            yi = score_map.view(b_s, l_s, -1)
            # print("[1] yi.shape: ",yi.shape)
            yi = F.softmax(yi*self.softmax_scale, dim=2).view(l_s, -1)
            pi = pi.view(h_s * w_s, -1)
            # print("pi.shape: ",pi.shape)
            yi = torch.mm(yi, pi)
            # print("[2] yi.shape: ",yi.shape)
            # print(self.stride_1)


            yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            zi = zi / out_mask
            y.append(zi)

        y = torch.cat(y, dim=0)
        y = self.W(y)
        y = b + y

        if self.add_SE:
            y_SE=self.SE(y)
            y=self.conv33(torch.cat((y_SE*y,y),dim=1))

        # print("y.shape: ",y.shape)
        return y

    def GSmap(self,a,b):
        return torch.matmul(a,b)

class SE_net(nn.Module):
    def __init__(self,in_channels,reduction=16):
        super(SE_net,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//reduction,kernel_size=1,stride=1,padding=0)
        self.fc2=nn.Conv2d(in_channels=in_channels//reduction,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        o1=self.pool(x)
        o1=F.relu(self.fc1(o1))
        o1=self.fc2(o1)
        return o1
class size_selector(nn.Module):
    def __init__(self,in_channels,intermediate_channels,out_channels):
        super(size_selector,self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=in_channels,out_features=intermediate_channels),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.selector_a = nn.Linear(in_features=intermediate_channels,out_features=out_channels)
        self.selector_b = nn.Linear(in_features=intermediate_channels, out_features=out_channels)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        vector = x.mean(-1).mean(-1)
        o1 = self.embedding(vector)
        a = self.selector_a(o1)
        b = self.selector_b(o1)
        v = torch.cat((a,b),dim=1)
        v = self.softmax(v)
        a = v[:,0,...,None,None,None]#.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = v[:,1,...,None,None,None]#.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # a = v[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # b = v[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return a,b
