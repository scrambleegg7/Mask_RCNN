import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project

ROOT_DIR = "/Users/donchan/Documents/Miyuki/Mask_RCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


from ShapeConfigClass import ShapesConfig

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("download mdodel")
    utils.download_trained_weights(COCO_MODEL_PATH)



def process():

    config = ShapesConfig()
    config.display()

    #loadmodel()

    model = modellib.MaskRCNN(mode="training", config=config,
                        model_dir=MODEL_DIR)

    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])



def main():
    process()



if __name__ == "__main__":
    main()