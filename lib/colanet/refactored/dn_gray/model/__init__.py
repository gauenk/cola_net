import os
from importlib import import_module
import torch
import torch as th
import torch.nn as nn
import numpy as np
import random

# -- modules --
from colanet.utils import clean_code
import colanet.utils.gpu_mem as gpu_mem
from . import adapt
from . import ca_forward

# try:
#     from colanet.utils import clean_code
#     from . import adapt
# except:
#     import model.adapt as adapt
#     def clean_code(*args,**kwargs): pass

# ------------------

def test_pad(model, L, modulo=16):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h / modulo) * modulo - h)
    paddingRight = int(np.ceil(w / modulo) * modulo - w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)
#     print(img_tensor.shape)
    return img_tensor.type_as(img)


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def test_x8(model, L, coords):
    E_list = [model(augment_img_tensor(L, mode=i),coords) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E

def test_x8_2(model, L):
    E_list = [augment_img_tensor(L, mode=i) for i in range(8)]
    list_input = torch.cat(E_list,dim=0)
    E_list = list(torch.chunk(model(list_input),list_input.shape[0],dim=0))
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E

@clean_code.add_methods_from(adapt)
@clean_code.add_methods_from(ca_forward)
class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.n_GPUs = args.n_GPUs

        self.save_models = args.save_models

        try:
            mname = 'colanet.refactored.dn_gray.model.'+args.model.lower()
            module = import_module(mname)
        except:
            module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)

        if not args.cpu:
            torch.cuda.manual_seed(args.seed)
            self.model.cuda()
            if args.precision == 'half':
                self.model.half()

            if args.n_GPUs > 1:
                gpu_list = range(0, args.n_GPUs)
                self.model = nn.DataParallel(self.model, gpu_list)

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        if args.print_model:
            print(self.model)

    def forward(self, x, idx_scale=0,ensemble=False,coords=None):
        self.ensemble = ensemble
        self.idx_scale = idx_scale
        # if self.ensemble:
        #     return test_x8(self.model, x, coords)
        # else:
        #     return self.model(x)#, coords)

        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        elif self.chop and not self.training:
            with torch.no_grad():  # this can save much memory
                Out = self.forward_chop(x,coords)
            return Out
        else:
            return self.model(x,coords)

    def my_fwd(self, x, ensemble=False):
        # return self.model(x)
        gpu_mem.reset_peak_gpu_stats()
        coords_l = self.get_coords_list(x.shape,128)
        y = th.zeros_like(x)
        for coords in coords_l:
            top,left,btm,right = coords
            yi = self.model(x,coords)
            # yi = self.model(x,None)#coords)
            zi = yi[...,top:btm,left:right]
            y[...,top:btm,left:right] = y[...,top:btm,left:right] + zi
        gpu_mem.print_peak_gpu_stats(True,"my_fwd",reset=True)
        return y

    def get_coords_list(self,vshape,bsize):
        t,c,h,w = vshape
        coords_l = []
        n_h = (h-1) // bsize + 1
        n_w = (w-1) // bsize + 1
        for hb in range(n_h):
            for wb in range(n_w):
                h_start,h_end = hb*bsize,(hb+1)*bsize
                w_start,w_end = wb*bsize,(wb+1)*bsize
                coords = [h_start,w_start,h_end,w_end]
                coords_l.append(coords)
        return coords_l

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_chop(self, x, coords=None, shave=10, min_size=10000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        #############################################
        # adaptive shave
        shave_scale = 4
        # max shave size
#         if self.ensemble == False:
#             shave_size_max=12
#         else:
        shave_size_max = 24
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        h_size = mod_h * shave_scale + shave_size_max
        w_size = mod_w * shave_scale + shave_size_max
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                # print("lr_batch.shape: ",lr_batch.shape)
                if self.ensemble == False:
                    sr_batch = self.model(lr_batch,coords)
                else:
                    sr_batch = test_x8(self.model, lr_batch, coords)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, coords=coords, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = torch.tensor(x.data.new(b, c, h, w))
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def adaptive_shave(self,h, w):

        # corresponding to scaling factor of the downscaling and upscaling modules in the network
        shave_scale = 4
        # max shave size
        shave_size_max = 24
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        midsize_h = mod_h * shave_scale + shave_size_max
        midsize_w = mod_w * shave_scale + shave_size_max
        # print('midsize_h={}, midsize_w={}'.format(midsize_h, midsize_w))
        return midsize_h, midsize_w

    '''
    def adaptive_shave(self, h, w):
        shave_scale = 4
        shave_size_max = 12
        h_half, w_half = h // 2, w // 2
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        midsize_h = mod_h * shave_scale + shave_size_max
        midsize_w = mod_w * shave_scale + shave_size_max

        return midsize_h, midsize_w
    '''
    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single':
                v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            if not self.cpu: ret = torch.Tensor(tfnp).cuda()
            if self.precision == 'half': ret = ret.half()

            return torch.tensor(ret)

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

