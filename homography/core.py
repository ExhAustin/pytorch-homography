import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
import torch
import torchvision

class DepthImgTransformer(object):
    """
    Base class for image transformations

        Camera transformation frame:
            +x axis - to the top of image
            +y axis - to the left of image
            +z axis - into the image
    """

    def __init__(self, K):
        # Camera intrinsics
        self.K = torch.from_numpy(K.astype('float32')).cuda()
        try:
            self.Kinv = torch.inverse(self.K)
        except ValueError:
            print("Error: Intrinsic matrix not invertible.")

    def transform(self, img, dx, dq, rgbd=True):
        """
        img: [width, height, num_channels] (num_channels = (4 if rgbd else 1))
        dx: camera translation
        dq: camera rotation in quaternion <w,x,y,z>
        rgbd: boolean value indicating whether color channels exist
        """
        imgs = np.array(img)[None,:]
        dxs = np.array(dx)[None,:]
        dqs = np.array(list(dq))[None,:]

        return self.transform_batch(imgs, dxs, dqs, rgbd)[0,:]

    def transform_batch(self, imgs, dxs, dqs, rgbd=True):
        """
        imgs: [N, width, height, num_channels] (num_channels = (4 if rgbd else 1))
        dxs: camera translations [N, 3]
        dqs: camera rotations in quaternion <w,x,y,z> [N, 4]
        rgbd: boolean value indicating whether color channels exist
        """
        N = imgs.shape[0]
        imgs = torch.from_numpy(imgs.astype('float32')).cuda()
        dxs = torch.from_numpy(np.array(dxs).astype('float32'))
        dqs = torch.from_numpy(np.array(dqs).astype('float32'))

        # Transformation matrix
        H = torch.zeros([N, 4, 4], dtype=torch.float32).cuda()
        H[:,0,0] = 1. - 2*(dqs[:,2]**2 + dqs[:,3]**2)
        H[:,0,1] = 2*(dqs[:,1]*dqs[:,2] - dqs[:,3]*dqs[:,0])
        H[:,0,2] = 2*(dqs[:,1]*dqs[:,3] + dqs[:,2]*dqs[:,0])
        H[:,1,0] = 2*(dqs[:,1]*dqs[:,2] + dqs[:,3]*dqs[:,0])
        H[:,1,1] = 1. - 2*(dqs[:,1]**2 + dqs[:,3]**2)
        H[:,1,2] = 2*(dqs[:,2]*dqs[:,3] - dqs[:,1]*dqs[:,0])
        H[:,2,0] = 2*(dqs[:,1]*dqs[:,3] - dqs[:,2]*dqs[:,0])
        H[:,2,1] = 2*(dqs[:,2]*dqs[:,3] + dqs[:,1]*dqs[:,0])
        H[:,2,2] = 1. - 2*(dqs[:,1]**2 + dqs[:,2]**2)
        H[:,0:3,3] = dxs
        H[:,3,3] = 1.

        return self._compute_homography(imgs, H)
