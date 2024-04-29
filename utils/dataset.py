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
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.gif'))

        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation['image']
            mask = augmentation['mask']
        
        return img, mask
    
    
    
