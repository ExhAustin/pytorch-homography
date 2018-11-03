import numpy as np
import torch

from .interpolation import BilinearInterp

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
    # Rollout pixel coordinates and values

    # Inverse transform pixel coordinates

    # Bilinear interpolation
