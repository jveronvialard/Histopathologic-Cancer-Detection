import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import PCamDataset
from resnet_binary_classifier import Resnet_Binary_Classifier

LEARNING_RATE = 5e-3
NUM_EPOCHS = 30
BATCH_SIZE = 50
SIZE_TRAIN_DATASET = 50 # To train on small dataset
SIZE_VAL_DATASET = 50

USE_GPU = True
#dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def main():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
    train_dataset = PCamDataset(csv_file="train_labels.csv", transform=data_transform)
    val_dataset = PCamDataset(csv_file="dev_labels.csv", transform=data_transform)
    
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
    
    model = Resnet_Binary_Classifier()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    epoch = 0
    while epoch != NUM_EPOCHS:
        epoch += 1
        for t, (x, y) in enumerate(loader_train):
            model.train()
            scores = model(x)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch %d, loss = %.4f' % (epoch, loss.item()))
        
        model.eval() # set model to evaluation mode
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for x, y in loader_train:
                x = x.to(device=device)  # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                preds = (scores > 0).type("torch.FloatTensor")
    
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
                preds = (scores > 0).type("torch.FloatTensor")
                num_correct += (preds==y).sum()
                num_samples += preds.size(0) 
            acc = float(num_correct) / num_samples
            print('Got accuracy %f: %d / %d on val set' % (acc, num_correct, num_samples))

if __name__ == "__main__":
    main()
