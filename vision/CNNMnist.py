#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, is_feat=False, preact=False):
        f1_pre = F.max_pool2d(self.conv1(x), 2)
        f1 = F.relu(f1_pre)

        f2_pre = F.max_pool2d(self.conv2_drop(self.conv2(f1)), 2)
        f2 = F.relu(f2_pre)
        
        x = f2
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
