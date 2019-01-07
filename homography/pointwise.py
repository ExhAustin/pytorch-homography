import numpy as np
from pyquaternion import Quaternion as Quat
import torch

from .core import DepthImgTransformer
from .utils.interpolation import PointInterp2d

class PointwiseHomographyTransformer(DepthImgTransformer):

    def _compute_homography(self, imgs0, H):
        img_dims = imgs0.shape[1:3]
        N = imgs0.shape[0]
        n_channels = imgs0.shape[3]

        imgs1 = torch.empty(imgs0.shape, dtype=torch.float32).cuda()

        # Create array of pixels (<batch, channels, pixels>)
        idc_mat = torch.empty(img_dims[0], img_dims[1], 3).cuda()
        idc_mat[:,:,0] = torch.arange(img_dims[0]).view(-1,1)
        idc_mat[:,:,1] = torch.arange(img_dims[1]).view(1,-1)
        idc_mat[:,:,2] = torch.ones(img_dims[0], img_dims[1])
        points_i0 = idc_mat.permute(2,0,1).view(1, 3, -1)
        points_v = imgs0.permute(0,3,1,2).view(N, n_channels, -1)

        # Compute projective transformation matrix (M = w0/w1*P)
        t_mat = torch.zeros(H[:,0:3,0:3].shape, dtype=torch.float32).cuda()
        t_mat[:,0:3,2] = H[:,0:3,3]
        x = torch.matmul(H[:,0:3,0:3], self.Kinv) + t_mat
        P = torch.matmul(self.K, x)

        # Transform pixels
        w0 = points_v[:,-1,:]
        points_i1 = w0.view(N,1,-1) * torch.matmul(P, points_i0)

        # Normalization & update depth
        w1 = points_i1[:,2,:]
        points_i1 = points_i1 / w1.view(N,1,-1)
        points_v[:,-1,:] = w1

        # Interpolation
        for k in range(N):
            interpolator = PointInterp2d(points_i1[k,0,:], points_i1[k,1,:], points_v[k,:,:])
            points_v1 = interpolator.query_batch(points_i0[:,0,:], points_i0[:,1,:])
            imgs1[k,:,:,:] = points_v1.view(img_dims[0], img_dims[1], channels)

        return imgs1
