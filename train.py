import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)
from config import Config
import utils
import model as modellib

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class BalloonConfig(Config):

    NAME = "shoe"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

