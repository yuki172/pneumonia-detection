from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import torch
import os
import time
from utils import get_pred_type
import numpy as np


def save_grad_cam(model, input_image, label, original_image, save_dir):

    with GradCAM(model=model, target_layers=[model.model.layer4[-1]]) as cam:
        grayscale_cam = cam(input_tensor=input_image.unsqueeze(0), targets=[ClassifierOutputTarget(0)])  # type: ignore
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            original_image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True
        )
        outputs = cam.outputs

        pred_label = torch.argmax(outputs, dim=1)[0].item()
        title = get_pred_type(label, pred_label)

        file_name = f"{title}-{time.time()}"
        # Plot
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.axis("off")
        # plt.show()
        plt.suptitle(title)

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{file_name}.png")
        plt.close(fig)
