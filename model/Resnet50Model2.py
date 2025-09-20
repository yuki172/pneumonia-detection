import torch
from torch import nn
from config import *
from torchvision.models import resnet50, ResNet50_Weights


class Resnet50Model2(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.requires_grad_(False)

        model.fc = nn.Sequential(  # type: ignore
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
        )

        self.model = model

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, x):
        return self.model(x)
