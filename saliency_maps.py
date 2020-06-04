# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 05:05:15 2020

@author: JulienVeronVialard
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from heapq import nlargest, nsmallest

from utils import PCamDataset


SAVE_MODEL_PATH = "./save/train_resnet18_sequential_unfreeze_best20200604T102208"


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


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    # Compute saliency
    N, _, H, W = X.shape
    scores = model.forward(X)    
    loss = scores.sum(axis=0)
    loss.backward()
    X_grad = X.grad
    saliency, _ = X_grad.abs().max(axis=1)

    return saliency


def main():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
    data_detransform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
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
    
    model = torch.load(ROOT_DIR + SAVE_MODEL_PATH.replace('.', ''))
    model.eval()
    

    with torch.no_grad():

        y_prob = torch.zeros(val_dataset.__len__())
        x_val = torch.zeros((val_dataset.__len__(), 3, 96, 96))
        y_val = torch.zeros(val_dataset.__len__())
        for t, (x, y) in enumerate(loader_val):
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            y_prob[t*BATCH_SIZE: min((t+1)*BATCH_SIZE, val_dataset.__len__())] = torch.sigmoid(scores).flatten()
            y_val[t*BATCH_SIZE: min((t+1)*BATCH_SIZE, val_dataset.__len__())] = y.flatten()
            x_val[t*BATCH_SIZE: min((t+1)*BATCH_SIZE, val_dataset.__len__()), :, :, :] = x

    # Saliency maps
    
    # True Positive
    mask = y_val
    n = 3
    tuples = nlargest(n, enumerate(y_prob*mask), key=lambda x: x[1])

    X = torch.stack([x_val[idx] for idx, prob in tuples])
    y = torch.stack([y_val[idx] for idx, prob in tuples])
    saliency = compute_saliency_maps(X, y, model)
    
    for i in range(n):
        saliency_im = transforms.ToPILImage(mode=None)(saliency[i].reshape(1, 96, 96))
        input_im = transforms.ToPILImage(mode=None)(data_detransform(x[i]))
        fig = plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(input_im)
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        plt.imshow(saliency_im)
        plt.axis('off')
        plt.show()


    # False Negative
    mask = y_val
    n = 3
    tuples = nsmallest(n, enumerate(y_prob*mask), key=lambda x: x[1])

    X = torch.stack([x_val[idx] for idx, prob in tuples])
    y = torch.stack([y_val[idx] for idx, prob in tuples])
    saliency = compute_saliency_maps(X, y, model)
    
    for i in range(n):
        saliency_im = transforms.ToPILImage(mode=None)(saliency[i].reshape(1, 96, 96))
        input_im = transforms.ToPILImage(mode=None)(data_detransform(x[i]))
        fig = plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(input_im)
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        plt.imshow(saliency_im)
        plt.axis('off')
        plt.show()        
    
    

if __name__ == "__main__":
    pass
    main()
    
    
    
    
