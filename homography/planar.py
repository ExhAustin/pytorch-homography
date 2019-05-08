import numpy as np
from pyquaternion import Quaternion as Quat
import cv2
import torch
import torchvision

from .core import DepthImgTransformer

class PlanarHomographyTransformer(DepthImgTransformer):

    def _compute_homography(self, imgs0, H, depth=None):
        img_dims = imgs0.shape[1:3]
        N = imgs0.shape[0]
        n_channels = imgs0.shape[3]

        imgs1 = torch.empty(imgs0.shape, dtype=torch.float32).cuda()

        # Estimate depth of entire image
        if depth is None:
            w_sum = torch.sum(imgs0[:,:,:,3], dim=(1,2))
            w_valid = torch.sum(imgs0[:,:,:,3]!=0, dim=(1,2), dtype=torch.float32)
            w0 = w_sum / w_valid
        else:
            w0 = depth
        dw = H[:,2,3] #assume orthogonal
        w1 = w0 + dw

        # Projective transformation matrix
        t_mat = torch.zeros(H[:,0:3,0:3].shape, dtype=torch.float32).cuda()
        t_mat[:,0:3,2] = H[:,0:3,3]
        x = torch.matmul(H[:,0:3,0:3], self.Kinv*w0.view(-1,1,1) + t_mat)
        M = torch.matmul(self.K, x) / w1.view(-1,1,1)

        # Transform image & update depth
        imgs1 = self.image_warp.warp_batch(imgs0, M)
        imgs1[:,:,:,3] = imgs1[:,:,:,3] + dw.view(-1,1,1) #assume orthogonal

        return imgs1
