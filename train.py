# -*- conding: utf-8 -*-
"""
train.py

script to load the dataset for training and testing

@author:
Kimariyb (kimariyb@163.com)

@license:
Licensed under the MIT License
For details, see the License file.

@Data:
2024/4/27 21:55
"""


import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from . import unet

# hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCH = 2
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "data/train/"