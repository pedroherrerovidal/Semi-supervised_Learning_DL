import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import init

import argparse
import json


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=4, stride=2),  
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 40, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 60, kernel_size=4, stride=2),  
            nn.ReLU(True),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 70, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(70)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(70, 60, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(60),
            nn.ConvTranspose2d(60, 40, kernel_size=5, stride=2),  
            nn.ReLU(True),
            nn.BatchNorm2d(40),
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            nn.ConvTranspose2d(20, 3, kernel_size=4, stride=2),  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
