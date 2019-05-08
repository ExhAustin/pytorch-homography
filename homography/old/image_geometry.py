import math

import numpy as np
from PIL import Image, ImageOps
#import matplotlib.pyplot as plt

class ImageProcessor(object):
    def __init__(self, img_dims, K, camera_height, window_dims):
        self.sess = None

        self.img_dims = img_dims

        # Parse camera intrinsics
        self.fx = K[0]
        self.fy = K[4]
        self.ppx = K[2]
        self.ppy = K[5]
        self.d0 = camera_height

        # Compute image patch dimensions
        U = np.abs(2.*(self.x2u_transform(0.5*np.array(window_dims))\
                - self.x2u_transform(np.zeros(2))))
        self.patch_dims = (int(U[0]), int(U[1]))

        # Other parameters
        d_padded = math.sqrt(img_dims[0]**2 + img_dims[1]**2)
        self.r_padded = math.ceil(d_padded/2)

    def get_patch_dims(self):
        return self.patch_dims

    def x2u_transform(self, X):
        if len(X.shape) > 1:
            U = np.zeros([X.shape[0], 2])
            U[:,0] = X[:,0] * self.fx / self.d0 + self.ppx
            U[:,1] = X[:,1] * self.fy / self.d0 + self.ppy
        else:
            U = np.zeros(2)
            U[0] = X[0] * self.fx / self.d0 + self.ppx
            U[1] = X[1] * self.fy / self.d0 + self.ppy

        return U

    def u2x_transform(self, U):
        if len(U.shape) > 1:
            X = np.zeros([U.shape[0], 2])
            X[:,0] = self.d0 * (U[:,0] - self.ppy) / self.fy
            X[:,1] = self.d0 * (U[:,1] - self.ppx) / self.fx
        else:
            X = np.zeros(2)
            X[0] = self.d0 * (U[0] - self.ppy) / self.fy
            X[1] = self.d0 * (U[1] - self.ppx) / self.fx

        return X

    def crop_image(self, rgb_imgs, depth_imgs, p_arr, dims):
        N = p_arr.shape[0]

        # Transform position to pixel coordinates
        U_arr = self.x2u_transform(p_arr[:,0:2])
        theta = -p_arr[:,2]*180./np.pi

        # Compute rotated coordinates
        u0_arr = U_arr[:,0] - self.ppx
        v0_arr = U_arr[:,1] - self.ppy
        cos_arr = np.cos(theta)
        sin_arr = np.sin(theta)
        u1_arr = (u0_arr*cos_arr + v0_arr*sin_arr + self.r_padded).astype(int)
        v1_arr = (u0_arr*sin_arr - v0_arr*cos_arr + self.r_padded).astype(int)

        # Crop images
        rgb_patches = np.zeros([N, dims[0], dims[1], 3])
        depth_patches = np.zeros([N, dims[0], dims[1], 1])
        for i in range(N):
            # Load images into PIL
            rgb_img = Image.fromarray(rgb_imgs[i,:])
            depth_img = Image.fromarray(depth_imgs[i,:])

            # Pad and rotate images
            rgb_padded = ImageOps.expand(rgb_img, int(self.r_padded-self.img_dims[1]/2))
            depth_padded = ImageOps.expand(depth_img, int(self.r_padded-self.img_dims[0]/2))

            rgb_rotated = rgb_padded.rotate(theta[i], resample=Image.BILINEAR)
            depth_rotated = depth_padded.rotate(theta[i], resample=Image.NEAREST)

            # Crop images
            box = [u1_arr[i]-math.floor(dims[0]/2), v1_arr[i]-math.floor(dims[1]/2),
                    u1_arr[i]+math.ceil(dims[0]/2+1e-4), v1_arr[i]+math.ceil(dims[1]/2+1e-4)]
            rgb_patch = rgb_rotated.crop(box)
            depth_patch = depth_rotated.crop(box)

            # Convert to numpy array
            rgb_patches[i,:] = np.asarray(rgb_patch)
            depth_patches[i,:] = np.asarray(depth_patch)[:,:,None]

        return rgb_patches, depth_patches
