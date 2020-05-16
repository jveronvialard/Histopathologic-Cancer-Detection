# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1scFu2q1h7aQrkQo_SG0LUJ42LaWussLq
"""

import torch
# assert '.'.join(torch.__version__.split('.')[:2]) == '1.4'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F  # useful stateless functions

import numpy as np

from utils import PCamDataset
from three_layers_convnet import ThreeLayerConvNet
from resnet_binary_classifier import Resnet_Binary_Classifier

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def check_accuracy_part34(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            preds = (scores > 0).type("torch.FloatTensor")

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part34(model, optimizer, loader_accuracy, epochs=1, print_every=100):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            scores = model(x)
            
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            #if t % print_every == 0:
            #    print('Iteration %d, loss = %.4f' % (t, loss.item()))
            #    check_accuracy_part34(loader_accuracy, model)
            #    print()
        print('Epoch %d, loss = %.4f' % (e, loss.item()))
        check_accuracy_part34(loader_accuracy, model)
        #print("Scores: ", scores)

train_dataset = PCamDataset(csv_file="train_labels.csv")
val_dataset = PCamDataset(csv_file="dev_labels.csv")

size = 100
train_subdataset = torch.utils.data.Subset(dataset=train_dataset, 
                        indices=np.random.choice(
                            a=np.arange(len(train_dataset)), 
                            size=size, 
                            replace=False))

count = 0
for t, (x, y) in enumerate(train_subdataset):
    if y.numpy()[0]==1.0:
        count += 1
print("% of 1: ", count / len(train_subdataset))

batch_size = 100
loader_train_dataset = train_subdataset
loader_val_dataset = val_dataset

loader_train = DataLoader(loader_train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)


loader_val = DataLoader(loader_val_dataset, 
                        batch_size=batch_size, 
                        shuffle=True)

epochs = 10
learning_rate = 5e-3
print_every = 10

in_channel = 3
channel_1 = 96
channel_2 = 48

model = Resnet_Binary_Classifier()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_part34(model, optimizer, loader_accuracy=loader_train, epochs=epochs, print_every=print_every)



from torchvision import datasets, models, transforms

