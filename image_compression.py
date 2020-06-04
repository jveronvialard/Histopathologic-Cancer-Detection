# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 06:13:34 2020

@author: JulienVeronVialard
"""


import torchvision.transforms as transforms

from io import BytesIO


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np


from utils import PCamDataset


ROOT_DIR = os.getcwd().replace("\\", "/")


# DATASET
BATCH_SIZE = 8 #  To train on small dataset
EVALUATE_EVERY = 4 #  To train on small dataset
SIZE_TRAIN_DATASET, SIZE_VAL_DATASET = 16, 16 #  To train on small dataset
#BATCH_SIZE = 256
# EVALUATE_EVERY = 50000
#SIZE_TRAIN_DATASET, SIZE_VAL_DATASET = None, None


# DEVICE SETUP
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = "torch.cuda.FloatTensor"
else:
    device = torch.device('cpu')
    dtype = "torch.FloatTensor"

print('using device:', device)


def imageJPEGCompression(array, qf=10):
    im = Image.fromarray(array)
    outputIoStream = BytesIO()
    im.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def main():
    data_transform = transforms.Compose([
        transforms.Lambda(imageJPEGCompression),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
        #transforms.ToPILImage(mode=None),
        #transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        #transforms.Grayscale(num_output_channels=3),
        # For data augmentation
        # transforms.RandomRotation(degrees=10, resample=False, expand=False, center=None, fill=None)
        # transforms.RandomVerticalFlip(p=1)
        # transforms.RandomHorizontalFlip(p=1)
    train_dataset = PCamDataset(csv_file="train_labels.csv", 
                                root_dir=ROOT_DIR+"/data/", 
                                transform=data_transform)
    val_dataset = PCamDataset(csv_file="dev_labels.csv", 
                              root_dir=ROOT_DIR+"/data/", 
                              transform=data_transform)
    
    if SIZE_TRAIN_DATASET is not None:
        train_dataset = torch.utils.data.Subset(dataset=train_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(train_dataset)), 
                            size=SIZE_TRAIN_DATASET, 
                            replace=False))
    if SIZE_VAL_DATASET is not None:
        val_dataset = torch.utils.data.Subset(dataset=val_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(train_dataset)), 
                            size=SIZE_VAL_DATASET, 
                            replace=False))
        
    loader_train = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

    
    loader_val = DataLoader(val_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True)




if __name__ == "__main__":
    pass
    main()
    
    
