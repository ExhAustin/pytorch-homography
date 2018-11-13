import time

import cv2
import numpy as np
from pyquaternion import Quaternion as Quat

from homography import PlanarHomographyTransformer

# Camera intrinsics
K = np.array([
    [618.9769897460938, 0.0, 324.6860046386719],
    [0.0, 618.9976196289062, 234.1637725830078],
    [0.0, 0.0, 1.0]])

if __name__ == '__main__':
    # Camera movement
    dx = [0.05,0.03,0.3]
    dq = Quat(axis=[0,0,1], angle=0.5)

    # Load image
    img = np.empty([480, 640, 4]).astype('float32')
    img[:,:,0:3] = cv2.imread("test_rgb1.png", flags=cv2.IMREAD_COLOR)
    img[:,:,3] = cv2.imread("test_depth1.png", flags=cv2.IMREAD_UNCHANGED) / 1000.

    # Transform
    start = time.time()

    transformer = PlanarHomographyTransformer(K)
    new_img = transformer.transform(img, dx, dq)

    end = time.time()
    print("Time elapsed: {} seconds.".format(end-start))

    # Visualize
    cv2.imshow('image', img[:,:,0:3].astype('uint8'))
    cv2.imshow('new_image', new_img[0,:,:,0:3].astype('uint8'))
    cv2.imshow('image_depth', (1000*img[:,:,3]).astype('uint8'))
    cv2.imshow('new_image_depth', (1000*new_img[0,:,:,3]).astype('uint8'))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
