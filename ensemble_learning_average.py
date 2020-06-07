# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 03:24:00 2020

@author: JulienVeronVialard
"""


import torch
#import torch.nn as nn
#import torch.optim as optim
from torch.utils.data import DataLoader
#import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import PCamDataset
#from resnet_binary_classifier import Resnet_Binary_Classifier, Resnet_Binary_Classifier_Sequential_Unfreeze
import os

from tqdm import tqdm
#from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import roc_auc_score
#import matplotlib.pyplot as plt



ROOT_DIR = os.getcwd().replace("\\", "/")
BATCH_SIZE = 1

SAVE_DIR = "./save/"

MODELS = [
    (SAVE_DIR + "train_resnet_sequential_unfreeze_best20200604T175157", 1),
    (SAVE_DIR + "train_resnet_sequential_unfreeze_best20200604T175157", 0),
]
SAVE_MODEL_PATH = SAVE_DIR + np.datetime64("now").astype(str).replace('-', '').replace(':', '')



# DEVICE SETUP
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = "torch.cuda.FloatTensor"
else:
    device = torch.device('cpu')
    dtype = "torch.FloatTensor"

print('using device:', device)


# MODELS LOADING
models = []
if device.type == 'cpu':
    for model_tuple in MODELS:
        path, weight = model_tuple
        model = (
            (torch.load(path, map_location=torch.device('cpu')), 
             weight)    
        )
        models.append(model)
else:
    for path, weight in MODELS:
        model = (
            (torch.load(path).cuda(), 
              weight)    
        )
        models.append(model)


def main():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
    test_dataset = PCamDataset(csv_file="test_labels.csv", 
                                root_dir=ROOT_DIR+"/data/", 
                                transform=data_transform)
        
    loader_test = DataLoader(test_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
    
    len_dataset = test_dataset.__len__()
    
    y_test = np.zeros(len_dataset)
    y_hat = np.zeros(len_dataset)
        
    with torch.no_grad():
        with tqdm(total=test_dataset.__len__()) as progress_bar:
            for t, (x, y) in enumerate(loader_test):
                y_test[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, len_dataset)] = y.numpy().reshape(len(y))
                # Update progress bar
                progress_bar.update(BATCH_SIZE)
                progress_bar.set_postfix(y="y_test")      
        np.save(SAVE_MODEL_PATH + "_y_test", y_test)
            
        
        for model, weight in models:
            with tqdm(total=test_dataset.__len__()) as progress_bar:
                for t, (x, y) in enumerate(loader_test):
                    y_hat[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, len_dataset)] += model(x).numpy().reshape(len(y)) * weight
                    # Update progress bar
                    progress_bar.update(BATCH_SIZE)
                    progress_bar.set_postfix(y="y_hat")    
        y_hat /= sum([weight for _, weight in models])
        np.save(SAVE_MODEL_PATH + "_y_hat", y_hat)
    

        
    
if __name__ == "__main__":
    main()