import datetime
import sys

import imageio
import numpy as np
from pyquaternion import Quaternion as Quat

from classification import network, parsers
from utils import image_geometry

img_dims = (480, 640)
camera_height = 0.723 #823?
K_camera = [
        618.9769897460938, 0.0, 324.6860046386719, 
        0.0, 618.9976196289062, 234.1637725830078, 
        0.0, 0.0, 1.0]
window_dims = (0.2, 0.2)
world2camera = [-0.031, 0.801, 0.737, 0.000, -0.381, 0.925, -0.000]
world2ee = [0.897, -0.801, -0.003, 0.707, 0.002, -0.707, 0.002]

def main():
    task_name = "ep30_bs100_lr1e-4_do0.5_hnorm0.15_cpcpcc_64_64"

    # Training parameters
    max_epochs = 20
    batchsize = 100
    validate_ratio = 0.05
    test_ratio = 0.05

    lr = 1e-4
    dropout = 0.5

    h_norm = 0.15

    # Data files
    #data_dir = "/media/hdd/efort_data/2018-09-25/" # isaac
    data_dir = "/home/dtroniak/.ros/" # deeperthought

    logfile_ls = [
            "2018-08-20-10-04-00StateMachine.log",
            "2018-08-20-17-39-00StateMachine.log",
            "2018-08-23-11-55-00StateMachine.log"]
    #logfile_ls = ["2018-08-20-17-39-00StateMachine.log"] # test dataset

    #----------------- Initialization
    # Model name
    now = datetime.datetime.now()
    task_name = task_name + "_{:02}{:02}-{:02}{:02}{:02}".format(
            now.month, now.day, now.hour, now.minute, now.second)

    # Parse data
    print("Parsing data files...")
    parsed_data = []
    for logfile in logfile_ls:
        print(logfile)
        data = parsers.efort_logfile_parser(data_dir + logfile)
        parsed_data = parsed_data + parsers.efort_data_parser(data, data_dir)
    M = len(parsed_data)
    print("{} entries of data parsed.".format(M))

    # Separate validation and test set
    idx_arr = np.arange(M)
    tv_ratio = validate_ratio + test_ratio

    tv_idcs = np.random.choice(idx_arr, int(tv_ratio*M))
    training_idcs = np.setdiff1d(idx_arr, tv_idcs)

    validation_idcs = np.random.choice(tv_idcs, int(validate_ratio*M))
    testing_idcs = np.setdiff1d(tv_idcs, validation_idcs)

    training_data = [parsed_data[idx] for idx in training_idcs]
    validation_data = [parsed_data[idx] for idx in validation_idcs]
    testing_data = [parsed_data[idx] for idx in testing_idcs]

    print("#training data: {}".format(len(training_data)))
    print("#validation data: {}".format(len(validation_data)))
    print("#test data: {}".format(len(testing_data)))

    # Initialize preprocessor
    processor = BatchProcessor(
            img_dims, camera_height, K_camera, window_dims, world2camera, h_norm)
    input_dims = processor.get_input_dims()
    print("Image input dims: ", input_dims)

    # Initialize network
    model = network.QualityNet(input_dims, lr)

    #----------------- Batch training
    print("\n----------------------------------------------------------------------------")
    print("Task name: {}".format(task_name))
    print("Training parameters:")
    print("\tmax_epochs={}".format(max_epochs))
    print("\tbatchsize={}".format(batchsize))
    print("\tlr={}".format(lr))
    print("\tdropout_keep_prob={}".format(dropout))

    print("Training network...")
    N = len(training_data)
    n_batches = int(np.ceil(float(N)/batchsize))
    idx_arr = np.arange(N)
    total_loss = 0
    try:
        for ie in range(max_epochs):
            # Validate
            rgb_batch, depth_batch, y = processor.batch_preprocess(validation_data)
            yhat = model.predict(rgb_batch, depth_batch)
            #print("y", y)
            #print("yhat", yhat)
            val_loss = -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))

            # Print info
            print("epoch={:3}\t loss={:.6f}\t validation_loss={:.6f}".format(
                ie, total_loss/n_batches, val_loss))

            # Train
            total_loss = 0
            for ib in range(n_batches):
                # Sample batch
                batch_idcs = np.random.choice(idx_arr, batchsize)
                raw_batch = [training_data[idx] for idx in batch_idcs]
                rgb_batch, depth_batch, y_batch = processor.batch_preprocess(raw_batch)

                # Train on batch
                loss = model.train(rgb_batch, depth_batch, y_batch, dropout=dropout)
                #loss = model.train(rgb_batch, depth_batch, np.ones(batchsize))
                total_loss += loss/batchsize
        
        print("\nTraining complete.\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted.\n")

    #----------------- Testing
    print("Testing network...")
    rgb_batch, depth_batch, y = processor.batch_preprocess(testing_data)
    yhat = model.predict(rgb_batch, depth_batch)
    loss = -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))
    np.save("models/"+task_name+"_y.npy", y)
    np.save("models/"+task_name+"_yhat.npy", yhat)

    print("Test loss={:.6f}".format(loss))

    #----------------- Save model
    print("Saving model...")
    model.save("models/"+task_name)

class BatchProcessor(object):
    def __init__(self, img_dims, camera_height, K, window_dims, world2camera, h_norm):
        self.img_dims = img_dims
        self.camera_height = camera_height
        self.h_norm = h_norm

        self.x_w2c = np.array(world2camera[0:3])
        self.q_w2c = Quat(world2camera[3:7])
        self.x_w2e = np.array(world2ee[0:3])
        self.q_w2e = Quat(world2ee[3:7])

        # Image transform object
        self.img_processor = image_geometry.ImageProcessor(
                img_dims, K, camera_height, window_dims)

        # Get input dims
        self.input_dims = self.img_processor.get_patch_dims()

    def get_input_dims(self):
        return self.input_dims

    def batch_preprocess(self, raw_batch):
        """
        Extract input vectors, load & stack images
        """
        N = len(raw_batch)

        # Initialize memory
        x_batch = np.zeros([N, 3])
        q_batch = np.zeros([N, 4])
        pc_batch = np.zeros([N, 3])
        raw_rgb_batch = (np.zeros((N,) + self.img_dims + (3,))).astype('uint8')
        raw_depth_batch = (np.zeros((N,) + self.img_dims)).astype('uint16')
        y_batch = np.zeros(N, dtype=np.float32)

        # Fill in data
        null_idcs = []
        for i in range(len(raw_batch)):
            # Get sampled pose
            x_batch[i,:] = raw_batch[i]['gr_pos']
            q_batch[i,:] = raw_batch[i]['gr_quat']

            # Get camera centric positions
            dx_e = x_batch[i,0:2] - self.x_w2e[2:0:-1]
            pc_batch[i,0] = 0.7071*dx_e[0] - 0.7071*dx_e[1]
            pc_batch[i,1] = -0.7071*dx_e[0] -0.7071*dx_e[1]
            #pc_batch[i,2] = (self.w2c_q * Quat(q_batch[i,:]) * Quat(0,0,0,1)).angle
            #pc_batch[i,2] = np.arcsin(Quat(q_batch[i,:]).rotate([0,0,1])[1])
            pc_batch[i,2] = (self.q_w2e * Quat(q_batch[i,:])).angle

            # Load images
            try:
                raw_rgb_batch[i,:] = imageio.imread(raw_batch[i]['pgp_rgb_file'])
                raw_depth_batch[i,:] = imageio.imread(raw_batch[i]['pgp_depth_file'])
            except ValueError:
                print("Warning: Cannot load image for no known reason.")
            #print(imageio.imread(raw_batch[i]['pgp_depth_file'])[235:245,315:325])

            # Assign labels
            y_batch[i] = raw_batch[i]['grasp_result']

        # Get rid of invalid entries
        for idx in null_idcs[::-1]:
            np.delete(raw_pc_batch, idx, axis=0)
            np.delete(raw_rgb_batch, idx, axis=0)
            np.delete(raw_depth_batch, idx, axis=0)
            np.delete(y_batch, idx, axis=0)

        # Crop images
        rgb_batch0, depth_batch0 = self.img_processor.crop_image(
                raw_rgb_batch, raw_depth_batch, pc_batch, self.input_dims)

        # Normalize
        rgb_batch1 = rgb_batch0/256.
        depth_batch1 = ((depth_batch0!=0.)*self.camera_height - depth_batch0/1000.) / self.h_norm
        rgb_batch = rgb_batch1.astype(np.float32)
        depth_batch = depth_batch1.astype(np.float32)

        #print("x_batch", x_batch[2,:])
        #print("pc_batch", pc_batch[2,:])

        #print("raw_rgb", raw_rgb_batch[2,235:245,315:325,0])
        #print("rbg_batch0", rgb_batch0[2,70:80,70:80,0])
        #print("rbg_batch1", rgb_batch1[2,70:80,70:80,0])
        #print("rbg_batch", rgb_batch[2,70:80,70:80,0])

        #print("raw_depth", raw_depth_batch[2,235:245,315:325])
        #print("depth_batch0", depth_batch0[2,70:80,70:80,0])
        #print("depth_batch1", depth_batch1[2,70:80,70:80,0])
        #print("depth_batch", depth_batch[2,70:80,70:80,0])

        return rgb_batch, depth_batch, y_batch

if __name__ == '__main__':
    main()
