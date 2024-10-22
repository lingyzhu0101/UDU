# -*- coding: utf-8 -*-
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
from .mobile_net_v2 import mobile_net_v2
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NIMA(nn.Module):
    def __init__(self, pretrained_base_model=False):
        super(NIMA, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


