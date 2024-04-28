# -*- conding: utf-8 -*-
"""
dataset.py

script to load the dataset for training and testing

@author:
Kimariyb (kimariyb@163.com)

@license:
Licensed under the MIT License
For details, see the License file.

@Data:
2024/4/27 21:55
"""

import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)