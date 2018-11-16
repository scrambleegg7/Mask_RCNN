import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.mySelect import miyukiCamera as mySelect




def process():


    print("ROOT DIR", ROOT_DIR)

    DATA_DIR = "/Users/donchan/Documents/Miyuki/MaskRCNN_data"
    # data is splitted under train and val
    #
    config = mySelect.MiyukiCameraConfig()
    MYSELECT_DIR = os.path.join(DATA_DIR, "datasets/mySelect")

    # Load dataset
    # Get the dataset from the releases page
    # https://github.com/matterport/Mask_RCNN/releases
    dataset = mySelect.MiyukiCameraDataset()
    dataset.load_classObjects(MYSELECT_DIR, "train")

    # Must call before using the dataset
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    #image_ids = np.random.choice(dataset.image_ids, 4)
    #print("4 x image_ids with randomized..", image_ids )

    image_id = 1
    mask, class_ids = dataset.load_mask(image_id)
    print("mask shape", mask.shape)
    print("class_ids", class_ids)    

def main():
    process()


if __name__ == "__main__":
    main()