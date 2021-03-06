#
# Object detector (by sequential file read from directory)
#
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
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

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

def detector(model,config, dataset, DATA_DIR):

    MRCNN_DATA_DIR = "/".join(  DATA_DIR.split('/')[:-1] )
    MRCNN_DATA_DIR = os.path.join( MRCNN_DATA_DIR, "mrcnn_image")
    print(MRCNN_DATA_DIR)
    
    images = glob( os.path.join(DATA_DIR, "*.jpg")   ) 

    print("* total length of images : ", len(images) )


    for f in images:

        print("Running on {}".format(f))
        # Read image
        image = skimage.io.imread(f)
        # Detect objects
        results = model.detect([image], verbose=1)
        r = results[0]

        print("- " * 40 )
        print("Scores --> ",  r['scores'])
        print("found Class Names --> ", [dataset.class_info[i]["name"]  for i in r['class_ids']] )  

        classes = [dataset.class_info[i]["name"]  for i in r['class_ids']]

        if "prescription" in classes:
            print("found prescription on %s" % f.split("/")[-1])
            image_file = f.split("/")[-1]
            shutil.copy( f,  os.path.join( MRCNN_DATA_DIR, image_file )  )





def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Sequential Reading File Object Detector.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/balloon/dataset",
                        help='Directory of the target dataset to detect')
    
    args = parser.parse_args()

    assert args.dataset ,\
            "Provide --image directory to apply detector"

    model, config = process()

    dataset = miyukiCamera.MiyukiCameraDataset()
    DATA_DIR = args.dataset

    detector(model, config, dataset, DATA_DIR)


if __name__ == "__main__":
    main()