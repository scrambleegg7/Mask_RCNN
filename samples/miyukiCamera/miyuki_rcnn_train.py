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

import json




import json
import datetime
import skimage.draw


# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
# define abosolute path
ROOT_DIR = "/home/donchan/Documents/Miyuki/Mask_RCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from samples.miyukiCamera import miyukiCamera

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("load mdodel")
    utils.download_trained_weights(COCO_MODEL_PATH)


DATA_DIR = "/home/donchan/Documents/Miyuki/MaskRCNN_data/datasets/miyukiCamera"

config = miyukiCamera.MiyukiCameraConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class MiyukiCameraDataset(utils.Dataset):


    def __init__(self, class_map=None):

        super(self.__class__, self).__init__(class_map)

        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("miyukiCamera", 1, "drug")
        self.add_class("miyukiCamera", 2, "prescription")
        self.add_class("miyukiCamera", 3, "hands")
        self.add_class("miyukiCamera", 4, "sheets")
        self.add_class("miyukiCamera", 5, "money")
        self.add_class("miyukiCamera", 6, "cointab")
        self.add_class("miyukiCamera", 7, "hair")

        self.add_class("miyukiCamera", 8, "documents")
        self.add_class("miyukiCamera", 9, "notes")
        self.add_class("miyukiCamera", 10, "envelope")
        self.add_class("miyukiCamera", 11, "insurance")
        self.add_class("miyukiCamera", 12, "receipt")
        

    def load_classObjects(self, dataset_dir, subset):
                
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        print("# load_classObjects(MiyukiCameraDataset) dataset_dir -> # ",dataset_dir)


        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            
            #polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [r['region_attributes'] for r in a['regions'].values()]

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 



            #print("class objects polygons : ", polygons)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "miyukiCamera",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                objects=objects)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.

        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        #print("## image info source ", image_info["source"])
        if image_info["source"] != "miyukiCamera":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8) # Generate the numbers of mask
        
        
        polygons_objects_id = []
        for i, p in enumerate(info["polygons"]):

            #############
            #              IMPORTANT 
            ############# 
            if p["name"] != "polygon":
                # if rect, skipped process to escape error from all_points_x/all_points_y
                continue
            #############
            #               
            ############# 


            #print("# index #", i, p["name"])
            #print(p)

            #
            #  keep only polygons object id from polygons json data..
            #
            polygons_objects_id.append(i)
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        names = []
        for i, a in enumerate( info["objects"] ):
            
            if i in polygons_objects_id:
                names.append(a["classification"])
                #print("info object of image_info : ", a)
            


        class_ids = np.array([self.class_names.index(n) for n in names])
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), class_ids.astype(np.int32)



    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "miyukiCamera":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)






"""Train the model."""
# Training dataset.
dataset_train = MiyukiCameraDataset()
dataset_train.load_classObjects(DATA_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = MiyukiCameraDataset()
dataset_val.load_classObjects(DATA_DIR, "val")
dataset_val.prepare()


## Load and display random samples
# if you want to do sanitary check for classification code whether it is allocated based on MiyukiCameraDataSet class
# PLEASE uncomment.

#image_ids = np.random.choice(dataset_train.image_ids, 4)
#for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

print("Create model in training mode")
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

print("start COCO model....")

                          # Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    print("Load weights trained on MS COCO, but skip layers that ")
    print("are different due to the different number of classes ")
    print("See README for instructions to download the COCO weights")

    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)