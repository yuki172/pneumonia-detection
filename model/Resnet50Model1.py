import torch
from torch import nn
from config import *
from torchvision.models import resnet50, ResNet50_Weights


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


class ResNet50BaseLayer(nn.Module):
    def __init__(self):
        super().__init__()

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.requires_grad_(False)

        # take the output of layer4 in https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        model.avgpool = nn.Identity()  # type: ignore
        model.fc = nn.Identity()  # type: ignore

        self.model = model

    def forward(self, x):
        return self.model.forward(x)


class Resnet50Model1(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.requires_grad_(False)

        # take the output of layer4 in https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        model.avgpool = nn.Identity()  # type: ignore
        model.fc = nn.Identity()  # type: ignore

        self.model = nn.Sequential(
            ResNet50BaseLayer(),
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
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, x):
        return self.model(x)
