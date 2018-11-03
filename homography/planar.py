import numpy as np
from pyquaternion import Quaternion as Quat
import cv2
import torch
import torchvision

from core import DepthImgTransformer
from .utils.interpolation import BilinearInterp

class PlanarHomographyTransformer(DepthImgTransformer):

    def _compute_homography(self, imgs0, H):
        img_dims = imgs0.shape[1:3]
        N = imgs0.shape[0]
        n_channels = imgs0.shape[3]

        imgs1 = torch.empty(imgs0.shape, dtype=torch.float32).cuda()

        # Estimate depth of entire image
        w_sum = torch.sum(imgs0[:,:,:,3], dim=(1,2)) 
        w_valid = torch.sum(imgs0[:,:,:,3]!=0, dim=(1,2), dtype=torch.float32)
        w0 = w_sum / w_valid
        w1 = w0 + H[:,2,3]

        # Projective transformation matrix
        t_mat = torch.zeros(H[:,0:3,0:3].shape, dtype=torch.float32).cuda()
        t_mat[:,0:3,2] = H[:,0:3,3]
        x = torch.matmul(H[:,0:3,0:3], self.Kinv) + t_mat
        M = (w0/w1).view(-1,1,1) * torch.matmul(self.K, x)

        # Transform image
        imgs = imgs0[0,:,:,:].squeeze().cpu().numpy()
        M_np = M[0,:,:].cpu().numpy()
        imgs1 = cv2.warpPerspective(imgs[:,:,0:3], M_np, dsize=(img_dims[1], img_dims[0]))

        return imgs1

        #return imgs1.cpu().numpy()
