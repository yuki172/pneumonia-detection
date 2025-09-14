import torch
from torch import nn
from .base_layer import BaseLayer
from config import *


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            BaseLayer(),
            # with input of shape (224, 224), layer4 of ResNet returns output of shape (2048, 7, 7). The final output flattens it
            # we bring the shape back and add conv layer on top of it
            Reshape((2048, 7, 7)),
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), padding=1),  # 7 + 2 - 2 = 7
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                1024, 1024, kernel_size=(3, 3), padding=1, stride=2
            ),  # (7 + 2 - 3) // 2 + 1 = 4
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),  # 4 + 2 - 2 = 4
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),  # 4 + 2 - 2 = 4
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 2048),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(2048, NUM_CLASSES),
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, x):
        return self.model(x)
