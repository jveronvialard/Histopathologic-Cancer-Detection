#import torch
import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader
#from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.models

#import numpy as np


class Resnet_Binary_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_layer = nn.AdaptiveAvgPool2d((224, 224))
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        # Freeze layers
        for param in self.resnet101.parameters():
            param.requires_grad = False
        self.resnet101.fc = nn.Linear(in_features=2048, out_features=1, bias=True)


    def forward(self, x):
        x_resized = self.resize_layer(x)
        scores = self.resnet101(x_resized)
        return scores
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class Resnet_to_2048(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_layer = nn.AdaptiveAvgPool2d((224, 224))
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        self.resnet101.fc = Identity()
        # Freeze layers
        for param in self.resnet101.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_resized = self.resize_layer(x)
        x_out = self.resnet101(x_resized)
        return x_out