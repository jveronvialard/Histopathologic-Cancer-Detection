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

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



ROOT_DIR = os.getcwd().replace("\\", "/")
BATCH_SIZE = 1

save_dir = "./save/"
SAVE_MODEL_PATH = save_dir + "train_resnet_sequential_unfreeze_best20200604T175157"

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
    test_dataset = PCamDataset(csv_file="test_labels.csv", 
                                root_dir=ROOT_DIR+"/data/", 
                                transform=data_transform)
        
    loader_test = DataLoader(test_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
    
    len_dataset = test_dataset.__len__()

    if device.type == 'cpu':
        model = torch.load(SAVE_MODEL_PATH, map_location=torch.device('cpu'))
    else:
        model = torch.load(SAVE_MODEL_PATH)
    
    if device.type == 'cuda':
        model.cuda()
    
    y_test = np.zeros(len_dataset)
    y_hat = np.zeros(len_dataset)
        
    model.eval()
    with torch.no_grad():
        for t, (x, y) in tqdm(enumerate(loader_test)):
            #print("Iteration {}/{}".format(t, len_dataset//BATCH_SIZE))
            y_test[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, len_dataset)] = y.numpy().reshape(len(y))
            y_hat[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, len_dataset)] = model(x).numpy().reshape(len(y))
    np.save("y_test", y_test)
    np.save("y_hat_resnet101_unfreeze", y_hat)
        
if __name__ == "__main__":
    pass
    #main()
