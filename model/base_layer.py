from torchvision.models import resnet50, ResNet50_Weights
from torch import nn


class BaseLayer(nn.Module):
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
