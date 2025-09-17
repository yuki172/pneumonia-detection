from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import torch
import os
import time


def save_grad_cam(model, image, label, save_dir):

    with GradCAM(model=model, target_layers=[model.model[0].model.layer4[-1]]) as cam:
        grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=[ClassifierOutputTarget(0)])  # type: ignore
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True
        )
        outputs = cam.outputs

        pred = torch.argmax(outputs, dim=1)[0]
        title = ""
        if pred == label:
            if label == 0:
                title = "Correct-Pneumonia"
            else:
                title = "Correct-Normal"
        else:
            if label == 0:
                title = "Incorrect-False-Negative"
            else:
                title = "Incorrect-False-Positive"

        file_name = f"{title}-{time.time()}"
        # Plot
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.axis("off")
        # plt.show()
        plt.suptitle(title)

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{file_name}.png")
        plt.close(fig)
