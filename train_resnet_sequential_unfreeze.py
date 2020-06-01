# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 02:10:16 2020

@author: JulienVeronVialard
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import PCamDataset
from resnet_binary_classifier import Resnet_Binary_Classifier, Resnet_Binary_Classifier_Sequential_Unfreeze
import os

from torch.utils.tensorboard import SummaryWriter


ROOT_DIR = os.getcwd().replace("\\", "/")
LEARNING_RATE = 5e-3
NUM_EPOCHS = 30
BATCH_SIZE = 64
#SIZE_TRAIN_DATASET = 64 # To train on small dataset
#SIZE_VAL_DATASET = 64
SIZE_TRAIN_DATASET, SIZE_VAL_DATASET = None, None
L2_WD = 1e-5
#LR_SCHEDULER_T_MAX = 10
#LR_SCHEDULER_ETA_MIN = 1e-5

# TENSORBOARD SETUP
save_dir = "./save/"
name = "train_resnet_sequential_unfreeze_" + np.datetime64("now").astype(str).replace('-', '').replace(':', '')
writer = SummaryWriter(save_dir+name)

# DEVICE SETUP
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = "torch.cuda.FloatTensor"
else:
    device = torch.device('cpu')
    dtype = "torch.FloatTensor"

print('using device:', device)

def main():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
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
    
    #model = Resnet_Binary_Classifier()
    model = Resnet_Binary_Classifier_Sequential_Unfreeze()
    
    if device.type == 'cuda':
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WD)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
    #                                                 T_max=LR_SCHEDULER_T_MAX, 
    #                                                 eta_min=LR_SCHEDULER_ETA_MIN)
    
    epoch = 0
    while epoch != NUM_EPOCHS:
        epoch += 1
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device)
            y = y.to(device=device)
            model.train()
            scores = model(x)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # TENSORBOARD
            # no smoothing
            running_loss  = loss.item()
            walltime = epoch*len(loader_train) + t*BATCH_SIZE
            writer.add_scalar('train/BCE', running_loss, walltime)
            
        print('Epoch %d, loss = %.4f' % (epoch, loss.item()))
                
        model.eval() # set model to evaluation mode
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for x, y in loader_train:
                x = x.to(device=device)  # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                preds = (scores > 0).type(dtype)
                preds = preds.to(device)
                num_correct += (preds==y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got accuracy %f: %d / %d on training set' % (acc, num_correct, num_samples))  
            num_correct = 0
            num_samples = 0
            for x, y in loader_val:
                x = x.to(device=device)  # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                preds = (scores > 0).type(dtype)
                preds = preds.to(device)
                num_correct += (preds==y).sum()
                num_samples += preds.size(0) 
            acc = float(num_correct) / num_samples
            print('Got accuracy %f: %d / %d on val set' % (acc, num_correct, num_samples))
    
        #scheduler.step()
        


if __name__ == "__main__":
    pass
    main()
