# -*- conding: utf-8 -*-
"""
train.py

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
from utils import loader
from models import unet

# hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCH = 2
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# dataset Directories
TRAIN_IMG_DIR = "./data/imgs/train"
TRAIN_MASK_DIR = "./data/masks/train"
VAL_IMG_DIR = "./data/imgs/val"
VAL_MASK_DIR = "./data/masks/val"
# image size
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

# set loss function and optimizer
train_losses = []
val_acc = []
val_dice = []

# set seed for reproducibility
seed = random.randint(1, 100)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_fn(loader, model, loss_fn, optimizer, scaler, device):
    """
    Function to train the model for one epoch
    
    Paramters
    ---------
    loader : DataLoader
        DataLoader for training data
    model : nn.Module
        Model to be trained
    loss_fn : nn.Module
        Loss function to be used
    optimizer : optim.Optimizer
        Optimizer to be used
    scaler : GradScaler
        GradScaler to be used for mixed precision training
    device : str
        Device to be used for training
    
    Returns
    -------
    float
        Average loss for the epoch
    """
    loop = tqdm(loader)
    total_losses = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device, dtype=torch.float32)
        targets = targets.unsqueeze(1).to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast():
            predict = model(data)
            loss = loss_fn(predict, targets)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_losses += loss.item()
        loop.set_postfix(loss=total_losses/(batch_idx+1))
        
    return total_losses / len(loader)


def get_transforms(is_rotate=True, is_flip=True):
    """
    Define transforms for data augmentation
    
    If is_rotate is True, the input image will be rotated randomly between -35 and 35 degrees.
    If is_flip is True, the input image will be flipped horizontally and/or vertically with a 50% probability.
    
    Parameters
    ----------
    is_rotate : bool, optional
        Whether to rotate the input image, by default True
    is_flip : bool, optional
        Whether to flip the input image, by default True
    
    Returns
    -------
        A.Compose
            A composition of transforms to be applied to the input data.
    """    
    transforms = []
    transforms.append(A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH))
    if is_rotate:
        transforms.append(A.Rotate(limit=35, p=1.0))
    if is_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.VerticalFlip(p=0.1))
    transforms.append(A.Normalize(
        mean=[0.0, 0.0, 0.0], 
        std=[1.0, 1.0, 1.0], 
        max_pixel_value=255.0,
    ))
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def check_accuracy(loader, model, device):
    """
    Function to check the accuracy of the model on the validation set
    
    Parameters
    ----------
    loader : DataLoader
        DataLoader for validation data
    model : nn.Module
        Model to be evaluated
    device : str
        Device to be used for evaluation
    
    Returns
    -------
    float
        Accuracy of the model on the validation set
    float
        Dice score of the model on the validation set
    """
    num_corrent = 0
    num_pixels = 0
    dice_socre = 0
    model.eval()
    
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device, dtype=torch.float32)
            targets = targets.unsqueeze(1).to(device=device, dtype=torch.float32)
            
            predict = torch.sigmoid(model(data))
            predict = (predict > 0.5).float()
            
            num_corrent += (predict == targets).sum()
            num_pixels += torch.numel(predict)
            
            dice_score += (2 * (predict * targets).sum()) / (2 * (predict * targets)).sum() + ((predict * targets) < 1).sum()
            
    accu = round(float(num_corrent) / num_pixels, 4)
    dice = round(float(dice_socre) / len(loader), 4)

    print(f'Got: {num_corrent}/{num_pixels} with accu: {accu:.4f}, dice: {dice:.4f}')
    print(f'Dice score: {dice_socre}/{len(loader)}')
    
    model.train()
    
    return accu, dice
    

def main():
    train_transforms = get_transforms(is_rotate=True, is_flip=True)
    val_transforms = get_transforms(is_rotate=False, is_flip=False)
    
    train_loader, val_loader = loader.get_loaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        valid_img_dir=VAL_IMG_DIR,
        valid_mask_dir=VAL_MASK_DIR,
        train_transform=train_transforms,
        valid_transform=val_transforms,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    
    model = unet.UNet(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    for index in range(NUM_EPOCH):
        print(f"Current epoch: {index+1}")
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler, DEVICE)
        train_losses.append(train_loss)
        
        accu, dice = check_accuracy(val_loader, model, DEVICE)
        val_acc.append(accu)
        val_dice.append(dice)
        
        print(f"Train Loss: {train_loss:.4f}, Val Acc: {accu:.4f}, Val Dice: {dice:.4f}")
    
    # save model
    torch.save(model.state_dict(), "unet.pth")
    
    # plot loss and accuracy curves
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_acc, label="val_acc")
    plt.plot(val_dice, label="val_dice")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()