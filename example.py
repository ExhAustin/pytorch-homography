import time

import cv2
import numpy as np
from pyquaternion import Quaternion as Quat

from homography import PointwiseHomographyTransformer, PlanarHomographyTransformer

K = np.array([
    [618.9769897460938, 0.0, 324.6860046386719],
    [0.0, 618.9976196289062, 234.1637725830078],
    [0.0, 0.0, 1.0]])

if __name__ == '__main__':
    start = time.time()

    # Params
    filename = "/.efort/data/run-2018-10-29-14-56-56/attempt-2018-10-29-15-07-16/2018-10-29-15-07-20"
    dx = [0.05,0.03,0.3]
    dq = Quat(axis=[0,0,1], angle=0.5)

    # Load image
    img = np.empty([480, 640, 4]).astype('float32')
    img[:,:,0:3] = cv2.imread(filename + "RGBImage.png", flags=cv2.IMREAD_COLOR)
    img[:,:,3] = cv2.imread(filename + "DepthImage.png", flags=cv2.IMREAD_UNCHANGED) / 1000.

    # Transform
    #transformer = PointwiseHomographyTransformer(K)
    transformer = PlanarHomographyTransformer(K)
    new_img = transformer.transform(img, dx, dq)

    end = time.time()
    print("Time elapsed: {} seconds.".format(end-start))

    # Visualize
    cv2.imshow('image', (1000*img[:,:,3]).astype('uint8'))
    cv2.imshow('new_image', (1000*new_img[0,:,:,3]).astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
