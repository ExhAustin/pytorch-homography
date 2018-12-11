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
    img[:,:,3] = cv2.inpaint(img[:,:,3], (img[:,:,3]==0.).astype('uint8'), 
            inpaintRadius=3, flags=cv2.INPAINT_NS)

    return img

def visualize(img, title=""):
    cv2.imshow(title+'_rgb', img[:,:,0:3].astype('uint8'))
    cv2.imshow(title+'_depth', (1000*img[:,:,3]).astype('uint8'))

if __name__ == '__main__':
    # Camera movement
    dx = [-0.1502,-0.2313, 0.465]
    dq = Quat(axis=[0,0,1], angle=np.pi*(9./8.)).elements
    #dx = [0.,0., 0.05]
    #dq = Quat(axis=[0,0,1], angle=0).elements

    dxs = np.stack([dx, dx], axis=0)
    dqs = np.stack([dq, dq], axis=0)

    # Load images
    data_dir = "/home/aswang/Desktop/efort_test/"
    img1 = load_image(data_dir+"a0_rgb.png", data_dir+"a0_depth.png")
    #img2 = load_image("test_rgb2.png", data_dir"1_depth.png")
    #imgs = np.stack([img1, img2], axis=0)

    # Transform
    start = time.time()

    transformer = PlanarHomographyTransformer(K)
    new_img = transformer.transform(img1, dx, dq)
    #new_imgs = transformer.transform_batch(imgs, dxs, dqs)

    end = time.time()
    print("Time elapsed: {} seconds.".format(end-start))

    # Visualize
    #visualize(img1, "img0")
    visualize(new_img, "img1")

    label_img = load_image(data_dir+"a1_rgb.png", data_dir+"a1_depth.png")
    visualize(label_img, "img2")
    #visualize(new_imgs[0,:], "img1")
    #visualize(new_imgs[1,:], "img2")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
