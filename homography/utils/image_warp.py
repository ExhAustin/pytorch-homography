import numpy as np
import torch

def warp_img(img, M):
    """
    Warps images
    """
    imgs = img.unsqueeze(0)
    M = M.unsqueeze(0)

    return warp_img_batch(img, M)

def warp_img_batch(imgs, M):
    """
    Warps a batch of images
    """
    img_dims = imgs.shape[1:3]
    img_size = img_dims[0]*img_dims[1]
    N = imgs.shape[0]
    n_channels = imgs.shape[3]

    idc_mat = torch.empty([img_dims[0], img_dims[1], 3], dtype=torch.float32).cuda()
    weights = torch.empty([N,img_size,4], dtype=torch.float32).cuda()
    values = torch.empty([N,img_size,n_channels,4], dtype=torch.float32).cuda()

    # Rollout pixel coordinates and values
    idc_mat[:,:,0] = torch.arange(img_dims[0]).view(-1,1)
    idc_mat[:,:,1] = torch.arange(img_dims[1]).view(1,-1)
    idc_mat[:,:,2] = torch.ones(img_dims[0], img_dims[1])
    points_i = idc_mat.permute(2,0,1).view(1, 3, -1)
    #points_v = imgs.permute(0,3,1,2).view(N, n_channels, -1)

    # Inverse transform pixel coordinates
    eye = M.new_ones(M.shape[-1]).diag().expand_as(M)
    Minv, _ = torch.gesv(eye, M)
    points_i0 = torch.matmul(Minv, points_i)
    points_i0[:,0,:] = torch.clamp(points_i0[:,0,:], 0, img_dims[0]-2)
    points_i0[:,1,:] = torch.clamp(points_i0[:,1,:], 0, img_dims[1]-2)

    # Bilinear interpolation
    i0_floor = torch.floor(points_i0)
    i0_ceil = i0_floor + 1

    weights[:,:,0] = (i0_ceil[:,0,:] - points_i0[:,0,:]) * (i0_ceil[:,1,:] - points_i0[:,1,:])
    weights[:,:,1] = (i0_ceil[:,0,:] - points_i0[:,0,:]) * (points_i0[:,1,:] - i0_floor[:,1,:])
    weights[:,:,2] = (points_i0[:,0,:] - i0_floor[:,0,:]) * (i0_ceil[:,1,:] - points_i0[:,1,:])
    weights[:,:,3] = (points_i0[:,0,:] - i0_floor[:,0,:]) * (points_i0[:,1,:] - i0_floor[:,1,:])

    i0_floor[:,2,:] = torch.arange(N).view(-1,1)
    i0_floor = i0_floor.type(torch.cuda.LongTensor)
    i0_ceil = i0_ceil.type(torch.cuda.LongTensor)
    values[:,:,:,0] = imgs[i0_floor[:,2,:], i0_floor[:,0,:], i0_floor[:,1,:], :]
    values[:,:,:,1] = imgs[i0_floor[:,2,:], i0_floor[:,0,:], i0_ceil[:,1,:], :]
    values[:,:,:,2] = imgs[i0_floor[:,2,:], i0_ceil[:,0,:], i0_floor[:,1,:], :]
    values[:,:,:,3] = imgs[i0_floor[:,2,:], i0_ceil[:,0,:], i0_ceil[:,1,:], :]

    points_v1 = torch.sum(weights.unsqueeze(2) * values, dim=3)
    imgs1 = points_v1.view(N, img_dims[0], img_dims[1], n_channels)

    return imgs1
