import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
import torch
import torchvision

from .utils.image_warp import ImageWarp

class DepthImgTransformer(object):
    """
    Base class for image transformations

        Image format (img):
             origin - top left corner
            +x axis - right
            +y axis - down

        Input camera transformation (dx, dq):
            +x axis - to the right of image
            +y axis - to the bottom of image
            +z axis - into the image
    """

    def __init__(self, K):
        # Camera intrinsics
        self.K = torch.from_numpy(K.astype('float32')).cuda()
        try:
            self.Kinv = torch.inverse(self.K)
        except ValueError:
            print("Error: Intrinsic matrix not invertible.")

        # Image warper
        self.image_warp = ImageWarp()

    def transform(self, img, dx, dq, rgbd=True, gpu=False, depth=None):
        """
        img: [width, height, num_channels] (num_channels = (4 if rgbd else 1))
        dx: camera translation
        dq: camera rotation in quaternion <w,x,y,z>
        rgbd: boolean value indicating whether color channels exist

        Note: input images are in cv2 format (top-left corner as origin)
        """
        imgs = np.array(img)[None,:]
        dxs = np.array(dx)[None,:]
        dqs = np.array(list(dq))[None,:]

        return self.transform_batch(imgs, dxs, dqs, rgbd, gpu)[0,:]

    def transform_batch(self, imgs, dxs, dqs, rgbd=True, gpu=False, depth=None):
        """
        imgs: [N, height, width, num_channels] (num_channels = (4 if rgbd else 1))
        dxs: camera translations [N, 3]
        dqs: camera rotations in quaternion <w,x,y,z> [N, 4]
        rgbd: boolean value indicating whether color channels exist

        Note: input images are in cv2 format (top-left corner as origin)
        """
        N = imgs.shape[0]

        # Convert format
        if gpu:
            #imgs = torch.flip(imgs.permute(0,2,1,3), dims=[1])
            imgs = imgs.permute(0,2,1,3)
        else:
            #imgs = np.flip(imgs.swapaxes(1,2), axis=1)
            imgs = imgs.swapaxes(1,2)

        # Preprocess to tensors
        if gpu:
            imgs = imgs.cuda()
            dxs = dxs.cuda()
            dqs = dqs.cuda()
        else:
            imgs = torch.from_numpy(imgs.astype('float32')).cuda()
            dxs = torch.from_numpy(np.array(dxs).astype('float32')).cuda()
            dqs = torch.from_numpy(np.array(dqs).astype('float32')).cuda()

        # Compute
        with torch.no_grad():
            # Build transformation matrix
            H = torch.zeros([N, 4, 4], dtype=torch.float32).cuda()
            H[:,0,0] = 1. - 2*(dqs[:,2]**2 + dqs[:,3]**2)
            H[:,0,1] = 2*(dqs[:,1]*dqs[:,2] + dqs[:,3]*dqs[:,0])
            H[:,0,2] = 2*(dqs[:,1]*dqs[:,3] - dqs[:,2]*dqs[:,0])
            H[:,1,0] = 2*(dqs[:,1]*dqs[:,2] - dqs[:,3]*dqs[:,0])
            H[:,1,1] = 1. - 2*(dqs[:,1]**2 + dqs[:,3]**2)
            H[:,1,2] = 2*(dqs[:,2]*dqs[:,3] + dqs[:,1]*dqs[:,0])
            H[:,2,0] = 2*(dqs[:,1]*dqs[:,3] + dqs[:,2]*dqs[:,0])
            H[:,2,1] = 2*(dqs[:,2]*dqs[:,3] - dqs[:,1]*dqs[:,0])
            H[:,2,2] = 1. - 2*(dqs[:,1]**2 + dqs[:,2]**2)
            H[:,0:3,3] = -dxs
            H[:,3,3] = 1.

            # Compute homography
            imgs_out = self._compute_homography(imgs, H, depth)

        # Convert format
        if gpu:
            #imgs_out = torch.flip(imgs_out, dims=[1]).permute(0,2,1,3)
            imgs_out = imgs_out.permute(0,2,1,3)
        else:
            imgs_out = imgs_out.cpu().numpy()
            #imgs_out = np.flip(imgs_out, axis=1).swapaxes(1,2)
            imgs_out = imgs_out.swapaxes(1,2)

        return imgs_out
