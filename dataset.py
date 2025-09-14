import torch
from config import *
from utils import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset, DataLoader
from typing import Literal
from torchvision import datasets
import os


class PneumoniaDetectionDataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "val", "test"],
        augment: bool = False,
        normalize: bool = False,
        max_samples: int | None = None,
    ):
        self.dataset = datasets.ImageFolder(
            os.path.join(DATA_PATH, split),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.classes = self.dataset.classes

        self.augment = augment
        self.normalize = normalize
        self.max_samples = max_samples

    def __getitem__(self, i):
        image, label = self.dataset[i]
        original_image = image

        # augment
        if self.augment:
            x_shift, y_shift, scale = get_random_shift_and_scale()
            hue, saturation = get_random_hue_and_saturation()
            angle = get_angle()
            image = get_augmented_image(
                image=image,
                x_shift=x_shift,
                y_shift=y_shift,
                angle=angle,
                scale=scale,
                hue=hue,
                saturation=saturation,
            )

        # normalize, https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        if self.normalize:
            image = TF.normalize(
                image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        return image, label, original_image

    def __len__(self):
        if self.max_samples != None:
            return self.max_samples
        return len(self.dataset)
