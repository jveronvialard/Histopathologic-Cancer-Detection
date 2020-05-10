import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import numpy as np

import torch.nn.functional as F  # useful stateless functions


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
                            out_features=num_classes,
                            bias=True)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print(x.shape, x)
        h = F.relu(self.conv2_layer1(x))
        h = F.relu(self.conv2_layer2(h))
        N = h.shape[0]
        scores = self.fc(h.view(N, -1))
        scores = F.sigmoid(scores)
        # scores = torch.squeeze(scores)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores
