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

def load_image(rgb_file, depth_file):
    img = np.empty([480, 640, 4]).astype('float32')
    img[:,:,0:3] = cv2.imread(rgb_file, flags=cv2.IMREAD_COLOR)
    img[:,:,3] = cv2.imread(depth_file, flags=cv2.IMREAD_UNCHANGED) / 1000.

    return img

def visualize(img):
    cv2.imshow('image', img[:,:,0:3].astype('uint8'))
    cv2.imshow('image_depth', (1000*img[:,:,3]).astype('uint8'))

if __name__ == '__main__':
    # Camera movement
    dx = [0.05,0.03,0.3]
    dq = Quat(axis=[0,0,1], angle=0.5)

    # Load images
    img1 = load_imgae("test_rgb1.png", "test_depth1.png")
    img1 = load_imgae("test_rgb2.png", "test_depth2.png")
    imgs = np.stack([img1, img2], axis=0)

    # Transform
    start = time.time()

    transformer = PlanarHomographyTransformer(K)
    #new_img = transformer.transform(img1, dx, dq)
    new_imgs = transformer.transform_batch(imgs, dx, dq)

    end = time.time()
    print("Time elapsed: {} seconds.".format(end-start))

    # Visualize
    visualize(imgs[0,:])
    visualize(imgs[1,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
