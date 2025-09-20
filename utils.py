from config import *
import random
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets


# from eda, the normal class in under-represented
def get_weighted_sampler(dataset):

    labels = torch.tensor([data[1] for data in dataset])
    class_count = torch.tensor([(labels == label).sum() for label in [0, 1]])
    weights = 1.0 / torch.tensor(
        [class_count[label] for label in labels], dtype=torch.float
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)  # type: ignore


def get_random_shift_and_scale():
    x_shift = int((2 * TRANSLATE * random.random() - TRANSLATE) * IMAGE_SIZE[0])
    y_shift = int((2 * TRANSLATE * random.random() - TRANSLATE) * IMAGE_SIZE[1])
    scale = 1 + SCALE * random.random()
    return x_shift, y_shift, scale


def get_random_hue_and_saturation():
    hue = 2 * HUE * random.random() - HUE
    saturation = SATURATION * random.random() + 0.9
    return hue, saturation


def get_angle():
    angle = 2 * ANGLE * random.random() - ANGLE
    return angle


def get_augmented_image(
    *, image, x_shift, y_shift, scale, angle, hue=0.0, saturation=1.0
):
    image = TF.affine(
        image, angle=angle, scale=scale, translate=[x_shift, y_shift], shear=[0.0]
    )
    image = TF.adjust_hue(image, hue)
    image = TF.adjust_saturation(image, saturation)
    return image


def get_run_name(
    model_name, epochs, optimizer_name, learning_rate, batch_size, augment, **kwargs
):
    return f"{model_name}-{optimizer_name.lower()}-{learning_rate}-{batch_size}-{epochs}-{'augment' if augment else 'no-augment'}"


import os
import shutil
import random


def create_val_split(val_count=350, seed=42):
    random.seed(seed)

    train_base = os.path.join(DATA_PATH, "train")
    val_base = os.path.join(DATA_PATH, "val")

    classes = ["0_PNEUMONIA", "1_NORMAL"]

    for class_name in classes:
        train_data_dir = os.path.join(train_base, class_name)
        val_data_dir = os.path.join(val_base, class_name)

        os.makedirs(val_data_dir, exist_ok=True)

        images = os.listdir(train_data_dir)

        val_images = random.sample(images, val_count)

        for img in val_images:
            src = os.path.join(train_data_dir, img)
            dst = os.path.join(val_data_dir, img)
            shutil.move(src, dst)


def get_pred_type(label, pred_label):
    if label == 0 and pred_label == 0:
        return TRUE_POSITIVE
    elif label == 0 and pred_label == 1:
        return FALSE_NEGATIVE
    elif label == 1 and pred_label == 1:
        return TRUE_NEGATIVE
    else:
        return FALSE_POSITIVE
