import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import PCamDataset
from resnet_binary_classifier import Resnet_Binary_Classifier, Resnet_to_2048
import os

ROOT_DIR = os.getcwd().replace("\\", "/")
BATCH_SIZE = 100
SIZE_TRAIN_DATASET = None # To train on small dataset
SIZE_VAL_DATASET = None
SIZE_TEST_DATASET = None
L2_WD = 1e-5
LR_SCHEDULER_T_MAX = 10
LR_SCHEDULER_ETA_MIN = 1e-5

USE_GPU = True
#dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def main():
    SIZE_TRAIN_DATASET = None # To train on small dataset
    SIZE_VAL_DATASET = None
    SIZE_TEST_DATASET = None
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
    test_dataset = PCamDataset(csv_file="test_labels.csv", 
                              root_dir=ROOT_DIR+"/data/", 
                              transform=data_transform)
    
    if SIZE_TRAIN_DATASET is not None:
        train_dataset = torch.utils.data.Subset(dataset=train_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(train_dataset)), 
                            size=SIZE_TRAIN_DATASET, 
                            replace=False))
    else:
        SIZE_TRAIN_DATASET = train_dataset.__len__()
        
    if SIZE_VAL_DATASET is not None:
        val_dataset = torch.utils.data.Subset(dataset=val_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(val_dataset)), 
                            size=SIZE_VAL_DATASET, 
                            replace=False))
    else:
        SIZE_VAL_DATASET = val_dataset.__len__()
        
    if SIZE_TEST_DATASET is not None:
        test_dataset = torch.utils.data.Subset(dataset=test_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(test_dataset)), 
                            size=SIZE_TEST_DATASET, 
                            replace=False))
    else:
        SIZE_TEST_DATASET = val_dataset.__len__()
    
        
    loader_train = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
    
    loader_val = DataLoader(val_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True)
    
    loader_test = DataLoader(test_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True)
    
    model = Resnet_to_2048()
    
    if device.type == 'cuda':
        model.cuda()
    
    X_train = np.zeros((SIZE_TRAIN_DATASET, 2048))
    y_train = np.zeros((SIZE_TRAIN_DATASET, 1))

    X_val = np.zeros((SIZE_VAL_DATASET, 2048))
    y_val = np.zeros((SIZE_VAL_DATASET, 1))

    X_test = np.zeros((SIZE_TEST_DATASET, 2048))
    y_test = np.zeros((SIZE_TEST_DATASET, 1))

        
    
    with torch.no_grad():
        for t, (x, y) in enumerate(loader_train):
            print("{}/{} training examples processed".format(t*BATCH_SIZE, SIZE_TRAIN_DATASET))
            x = x.to(device=device)  # move to device, e.g. GPU
            X_train[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, SIZE_TRAIN_DATASET), :] = model(x).cpu().numpy()
            y_train[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, SIZE_TRAIN_DATASET), :] = y.numpy()
            np.save("X_train.npy", X_train)
            np.save("y_train.npy", y_train)
            
        
        for t, (x, y) in enumerate(loader_val):
            print("{}/{} validation examples processed".format(t*BATCH_SIZE, SIZE_VAL_DATASET))
            x = x.to(device=device)  # move to device, e.g. GPU
            X_val[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, SIZE_VAL_DATASET), :] = model(x).cpu().numpy()
            y_val[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, SIZE_VAL_DATASET), :] = y.numpy()
            np.save("X_val.npy", X_val)
            np.save("y_val.npy", y_val)
            
        for t, (x, y) in enumerate(loader_test):
            print("{}/{} validation examples processed".format(t*BATCH_SIZE, SIZE_TEST_DATASET))
            x = x.to(device=device)  # move to device, e.g. GPU
            X_test[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, SIZE_TEST_DATASET), :] = model(x).cpu().numpy()
            y_test[t*BATCH_SIZE:min((t+1)*BATCH_SIZE, SIZE_TEST_DATASET), :] = y.numpy()
            np.save("X_test.npy", X_test)
            np.save("y_test.npy", y_test)

if __name__ == "__main__":
    main()

    
    



