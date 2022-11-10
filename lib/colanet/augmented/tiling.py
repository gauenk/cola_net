
# -- imports --
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
