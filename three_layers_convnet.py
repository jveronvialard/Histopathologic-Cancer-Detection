#import torch
import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader
#from torch.utils.data import sampler
import torch.nn.functional as F

#import numpy as np


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2):
        super().__init__()
        self.conv2_layer1 = nn.Conv2d(in_channels=in_channel,
                                      out_channels=channel_1,
                                      kernel_size=(5, 5),
                                      stride=1,
                                      padding=2,
                                      bias=True)
        nn.init.kaiming_normal_(self.conv2_layer1.weight)
        nn.init.zeros_(self.conv2_layer1.bias)

        self.conv2_layer2 = nn.Conv2d(in_channels=channel_1,
                                      out_channels=channel_2,
                                      kernel_size=(3, 3),
                                      stride=1,
                                      padding=1,
                                      bias=True)
        nn.init.kaiming_normal_(self.conv2_layer2.weight)
        nn.init.zeros_(self.conv2_layer2.bias)

        self.fc = nn.Linear(in_features=channel_2 * 96 * 96,
                            out_features=1,
                            bias=True)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        h = F.relu(self.conv2_layer1(x))
        h = F.relu(self.conv2_layer2(h))
        N = h.shape[0]
        scores = self.fc(h.view(N, -1))
        
        return scores
