"""
Mask R-CNN
Configurations and data loading code for RSNA Pneumonia dataset.

Copyright (c) 2020 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 rsna.py train --dataset=/path/to/rsna/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 rsna.py train --dataset=/path/to/rsna/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 rsna.py train --dataset=/path/to/rsna/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 rsna.py train --dataset=/path/to/rsna/ --model=last

    # Run COCO evaluation on the last model you trained
    python3 rsna.py evaluate --dataset=/path/to/rsna/ --model=last
"""

import os
import sys
import time
import numpy as np
import skimage
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pydicom as dicom
import random
import keras
import math
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import imgaug.augmenters as iaa

from keras.callbacks import Callback
from mrcnn import visualize
from mrcnn.model import log
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Import custom utils
from rsna_utils import RsnaUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# TODO: update the path
RSNA_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_rsna.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    
############################################################
#  Configurations
############################################################

# This function keeps the learning rate at 0.0001 
# for the first five epochs and decreases it 
# exponentially after that.
def scheduler(epoch):
    print("Epoch:", epoch)
    if epoch < 5:
        return 0.0001
    else:
        return 0.0001 * math.pow(math.e, 0.1 * (5 - epoch))
    
# TODO: update the value
CUSTOM_LR = 0

class RsnaConfig(Config):
    """Configuration for training on RSNA Pneumonia.
    Derives from the base Config class and overrides values specific
    to the RSNA dataset.
    """
    # Give the configuration a recognizable name
    NAME = "rsna"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Uncomment to train on 3 GPUs (default is 1)
    GPU_COUNT = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + 1 class (pneumonia)
    
    # Learning rate and momentum
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9
    
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    
############################################################
#  Dataset
############################################################

class RsnaDataset(utils.Dataset):
    def train_val_dataset(self, df1, df2, df, images_dir):
        """
        Load the train/val dataset.
        df1: Target == 0 dataframe
        df2: Target == 1 dataframe
        """
        for patientId in df1.patientId.unique():
            imageId = patientId
            imgPath = os.path.join(images_dir, patientId) + ".png"
            h, w = 1024, 1024
            ann = list()
            self.add_image("rsna", image_id = imageId, path = imgPath, annotation = ann, height=h, width=w)
        for patientId in df2.patientId.unique():
            imageId = patientId
            imgPath = os.path.join(images_dir, patientId) + ".png"
            h, w = 1024, 1024
            ann = df[df.patientId.str.match(patientId)]
            self.add_image("rsna", image_id = imageId, path = imgPath, annotation = ann, height=h, width=w)
            
    def load_rsna(self, dataset_dir, subset, df1, df2):
        """Load a subset of the RSNA dataset.
        dataset_dir: The root directory of the RSNA dataset.
        subset: What to load (train or val)
        """
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        
        rsna_utils = RsnaUtils(dataset_dir)
        
        # Load annotations and sort by Target column
        df = rsna_utils.prepare_label()
        
        # Images directory
        images_dir = os.path.join(dataset_dir, "train")
                                      
        # Add class
        self.add_class("rsna", 1, "pneumonia")
        
        # Add images
        self.train_val_dataset(df1, df2, df, images_dir)
                  
    def extract_boxes(self, x, y, w, h):
        boxes = list()
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        boxes.extend([x, y, w, h])
        return boxes

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Get image info
        info = self.image_info[image_id]
        
        # Get image annotation
        ann = info['annotation']
        
        # For Target == 0
        if len(ann) == 0:
            boxes = list()  
        # For Target == 1 
        else:
            boxes = [self.extract_boxes(x, y, w, h) for x, y, w, h in 
                     zip(ann['x'], ann['y'], ann['width'], ann['height'])]
        # Create mask
        mask = np.zeros([info['height'], info['width'], len(boxes)], dtype=np.uint8)
        for i in range(len(boxes)):
            box = boxes[i]
            # Row is in fact Column and vice-versa
            row_s, row_e = box[1], box[1] + box[3]
            col_s, col_e = box[0], box[0] + box[2]
            mask[row_s:row_e, col_s:col_e, i] = 1
            
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == "rsna":
            return info['path']
        else:
            return "You tell me"
    
    def image_info_alt(self, image_id):
        """
        Return the image info
        """
        return self.image_info[image_id]

def train(model, model_path):
    """Train the model."""
    assert model_path != ""
    
    # Update the value if needed
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_rsna.h5")
    
    # Image Augmentation
    # GaussianBlur
    # Sharpen
    # Change brightness
    # Change contrast
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=0.3, name="reduce-noise"),
        iaa.Sharpen(alpha=0.75, lightness=0.70, name="crispier-xrays"),
        iaa.Multiply(1.15, name="bright-xrays"),
        iaa.GammaContrast(1.2, name="gamma-xrays")
    ], name="chest-xrays-aug")
    print("Image augmentation:", seq)
    
    dataset_dir = os.path.join(ROOT_DIR, "datasets/rsna")
    
    rsna_utils = RsnaUtils(dataset_dir)
    
    X_train, X_val, Y_train, Y_val = rsna_utils.prepare_rsna()
    
    # Training dataset.
    print("Loading the training dataset...")
    dataset_train = RsnaDataset()
    dataset_train.load_rsna(args.dataset, "train", X_train, Y_train)
    dataset_train.prepare()
    print("Train:", len(dataset_train.image_ids))
    
    # Random seed
    random.seed(datetime.now())
    
    X_val, _ = train_test_split(X_val, test_size=0.5, random_state=random.randint(0, 69))
    Y_val, _ = train_test_split(Y_val, test_size=0.5, random_state=random.randint(0, 69))

    # Validation dataset
    print("Loading the validation dataset...")
    dataset_val = RsnaDataset()
    dataset_val.load_rsna(args.dataset, "val", X_val, Y_val)
    dataset_val.prepare()
    print("Val:", len(dataset_val.image_ids))
    
    # Custom callbacks
    custom_callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto',
            cooldown=0, min_lr=0
        ),
        keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    ]
    
    # *** This training schedule is an example. Update to your needs. In case of OOM in stage 3, comment out the other two ***
    
    # Training - Stage 1
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # print("Training head layers...")
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads', augmentation=seq, custom_callbacks=custom_callbacks)
    
    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
     
    model.keras_model.save_weights(model_path)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=70,
                layers='4+', augmentation=seq, custom_callbacks=custom_callbacks)
    
    model.keras_model.save_weights(model_path)

    # Training - Stage 3
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='all', augmentation=seq, custom_callbacks=custom_callbacks)
    
    model.keras_model.save_weights(model_path)
    
    # Have a nice day!
    print("Have a nice day!")
    
        
############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect pneumonia on chest X-rays.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'infer'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/rsna/",
                        help='Directory of the RSNA dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = RsnaConfig()
    else:
        class InferenceConfig(RsnaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "rsna":
        weights_path = RSNA_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        model_path = os.path.join(MODEL_DIR, "mask_rcnn_rsna.h5")
        train(model, model_path)
    elif args.command == "infer":
        # do smt
        print("Something for now")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
        
