import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as TF
from grad_cam import compute_grad_cam


class PneumoniaModel:
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

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_image = self.transform(image)
        input_image = TF.normalize(
            original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # type: ignore
        ).to(self.device)
        visualization, pred_label = compute_grad_cam(
            self.model, input_image, original_image
        )

        return visualization, pred_label, original_image
