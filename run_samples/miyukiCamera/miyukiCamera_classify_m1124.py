#
# Object detector (by sequential file read from directory)
#

import cv2
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
import argparse
import skimage
import shutil
import skimage

from cv2 import imshow

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

#from samples.cats_dogs import cats_dogs
from samples.miyukiCamera import miyukiCamera


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")




from builtins import FileExistsError
import logging, sys, os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def process():

    class InferenceConfig(miyukiCamera.MiyukiCameraConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    #config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    # set model
    # Create model in inference mode
    with tf.device(DEVICE):    
        model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=MODEL_DIR)
    
    # Or, load the last model you trained
    weights_path = model.find_last()
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return model, config




def videopipeline():

    model, config = process()
    dataset = miyukiCamera.MiyukiCameraDataset()
    #DATA_DIR = args.dataset


    # AXIS M1124 Video streaming



    cam = cv2.VideoCapture()
    cam.open("http://192.168.1.151/axis-cgi/mjpg/video.cgi?fps=1")

    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)

    #
    # open SSDPipeline class
    #




    parentDir = "./results"

    prv_frame = None
    while(True):

        ret, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prv_frame is None:
            prv_frame = gray
            continue

        frameDelta = cv2.absdiff(prv_frame, gray)
        # if frameDelta = difference less than 30, black,
        # frameDelta bigger than 30, then white
        thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
        
        results = model.detect([frame], verbose=1)
        r = results[0]

        print("- " * 40 )
        print("Scores --> ",  r['scores'])
        print("found Class Names --> ", [dataset.class_info[i]["name"]  for i in r['class_ids']] )  

        classes = [dataset.class_info[i]["name"]  for i in r['class_ids']]
        if len(classes) > 0:

            if "Prescription" in classes:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                MMDDHH = time.strftime("%m%d%H")

                subdir = os.path.join( parentDir, MMDDHH)
                try:
                    os.makedirs(subdir)
                except FileExistsError as e:
                    pass

                _num = cv2.countNonZero(thresh)
                if _num > 10000:
                    logging.info("frame diffs happened bigger than threshhold --> %d" % _num )
                    logging.info("Name %s" % (classes,) )
                    #logging.info("Probability %s" % (probs,)   )
                    cv2.imwrite(  os.path.join( subdir, 'prescription-%s.jpg' % timestr )    , frame )
                
        prv_frame = gray

        #imshow("frameDelta", frameDelta)
        #imshow("thresh", thresh)
        resize_frame = skimage.transform.resize(frame, (300,480) )
        imshow("marked", resize_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     




def main():
    
    videopipeline()



if __name__ == "__main__":
    main()