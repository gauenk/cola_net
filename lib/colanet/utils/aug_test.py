import torch as th
import numpy as np

def test_x8(model, vid, flows=None, use_refine=False):

    # -- forward process --
    E_list = []
    inds_save = None
    for i in range(8):
        vid_aug,inds_aug = augment_img_tensor(vid, inds_save, mode=i)
        vid_e = model(vid_aug,flows,inds_aug)
        if (i == 0) and use_refine: inds_save = model.inds_buffer
        E_list.append(vid_e)

    # -- reverse augmentation --
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i],_ = augment_img_tensor(E_list[i], None, mode=8 - i)
        else:
            E_list[i],_ = augment_img_tensor(E_list[i], None, mode=i)

    # -- stack and normalizae --
    output_cat = th.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E

def augment_img_tensor(img, inds, mode=0):
    device = img.device
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))

    # -- aug img --
    img_np = augment_img(img_np, mode=mode)
    img_tensor = th.from_numpy(np.ascontiguousarray(img_np))

    # -- aug img --
    if not(inds is None):
        inds_np = inds.cpu().numpy()
        inds_np = augment_inds(inds_np, img_np.shape, mode=mode)
        inds = th.from_numpy(np.ascontiguousarray(inds_np))
        inds = inds.to(device)

    # -- shape --
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)
    return img_tensor.type_as(img),inds

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

def augment_inds(inds, vshape, mode=0):
    # if not(inds is None):
    #     print("inds.shape: ",inds.shape)
    if inds is None: return inds
    return inds
    # -- shape to img --
    H,W,C,T = vshape
    nH,nW = (H-1)//4+1,(W-1)//4+1
    ishape = inds.shape
    N,_,Q,K,_ = inds.shape
    inds = inds.reshape((N,1,T,nH,nW,K,3))
    inds = inds.transpose((3,4,0,1,2,5,6))

    # -- aug --
    if mode == 0:
        pass
    elif mode == 1:
        inds = np.flipud(np.rot90(inds))
    elif mode == 2:
        inds = np.flipud(inds)
    elif mode == 3:
        inds = np.rot90(inds, k=3)
    elif mode == 4:
        inds = np.flipud(np.rot90(inds, k=2))
    elif mode == 5:
        inds = np.rot90(inds)
    elif mode == 6:
        inds = np.rot90(inds, k=2)
    elif mode == 7:
        inds = np.flipud(np.rot90(inds, k=3))

    # -- shape back --
    inds = inds.transpose(np.argsort((3,4,0,1,2,5,6)))
    inds = inds.reshape(ishape)

    return inds

