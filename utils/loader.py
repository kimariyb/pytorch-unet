# -*- conding: utf-8 -*-
"""
loader.py

@author:
Kimariyb (kimariyb@163.com)

@license:
Licensed under the MIT License
For details, see the License file.

@Data:
2024/4/27 21:55
"""
import torch
import torchvision
import matplotlib.pyplot as plt

from dataset import CarvanaDataset
from torch.utils.data import DataLoader



def get_loaders(
    train_img_dir, train_mask_dir, valid_img_dir, valid_mask_dir, 
    train_transform, valid_transform, batch_size, num_workers, pin_memory=True):
    """
    Get data loader for training and validation.

    Parameters
    ----------
    train_img_dir : str
        Path to the directory containing training images.
    train_mask_dir : str    
        Path to the directory containing training masks.
    valid_img_dir : str
        Path to the directory containing validation images.
    valid_mask_dir : str
        Path to the directory containing validation masks.
    train_transform : torchvision.transforms
        Transformations to apply to the training images.
    valid_transform : torchvision.transforms
        Transformations to apply to the validation images.
    batch_size : int
        Batch size for the data loader.
    num_workers : int
        Number of workers to use for the data loader.
    pin_memory : bool, optional
        Whether to pin memory for the data loader, by default True.

    Returns
    -------
    train_loader : DataLoader
        Data loader for training.
    valid_loader : DataLoader
        Data loader for validation.
    """
    train_dataset = CarvanaDataset(train_img_dir, train_mask_dir, train_transform)
    valid_dataset = CarvanaDataset(valid_img_dir, valid_mask_dir, valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, valid_loader