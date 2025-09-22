import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as TF
from grad_cam import compute_grad_cam
from typing import Tuple
import numpy as np


class PenumoniaClassifier:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = torch.load(model_path, weights_only=False, map_location=device)
        self.model.eval().to(device)

        # compute grad for grad cam
        for param in self.model.parameters():
            param.requires_grad = True

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, int, np.ndarray]:
        original_image = self.transform(image)
        input_image = TF.normalize(
            original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # type: ignore
        ).to(self.device)
        visualization, pred_label = compute_grad_cam(
            self.model, input_image, original_image
        )

        np_original_image = original_image.permute(1, 2, 0).numpy()  # type: ignore

        return visualization, pred_label, np_original_image  # type: ignore
