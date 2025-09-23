import torch
from torch import nn
from config import *
from torchvision.models import resnet50, ResNet50_Weights


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        model = nn.Sequential(  # type: ignore
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, padding=0),  # (224 + 0 - 2) // 2 + 1 = 112
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2, padding=0),  # (112 + 0 - 2) // 2 + 1 = 112
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2, padding=0),  # (56 + 0 - 2) // 2 + 1 = 28
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2, padding=0),  # (28 + 0 - 2) // 2 + 1 = 14
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, NUM_CLASSES),
        )

        self.model = model

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, x):
        return self.model(x)
