import argparse
import torch
from model.DetectionModel import DetectionModel
from dataset import PneumoniaDetectionDataset
from tqdm import tqdm
import os
import shutil
import json
from config import *
from torch import nn
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from utils import *
from grad_cam import save_grad_cam
from typing import Dict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=int,
    dest="model_path",
    help="model path",
)
parser.add_argument(
    "--batch_size",
    type=int,
    dest="batch_size",
    help="batch size",
)
parser.add_argument(
    "--max_samples",
    type=int,
    dest="max_samples",
    help="max samples",
)


parser.add_argument(
    "--save_folder",
    type=int,
    dest="save_path",
    help="save path",
)

GRAD_CAM_COUNT = 4


def test(model_path, save_folder, batch_size, max_samples):
    for name, arg in [["save_folder", save_folder], ["model_path", model_path]]:
        print(name, arg)

    model = torch.load(model_path, weights_only=False)
    test_set = PneumoniaDetectionDataset(split="test", augment=False, normalize=True)
    if max_samples != None:
        test_set = torch.utils.data.Subset(test_set, range(max_samples))

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device", device)

    model.to(device)

    metrics: Dict[str, float | None] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    grad_cam = {
        TRUE_POSITIVE: [],
        TRUE_NEGATIVE: [],
        FALSE_POSITIVE: [],
        FALSE_NEGATIVE: [],
    }
    met_count = 0

    accuracy_metric = MulticlassAccuracy(num_classes=2, average="micro")
    precision_metric = MulticlassPrecision(num_classes=2, average=None)
    recall_metric = MulticlassRecall(num_classes=2, average=None)
    f1_metric = MulticlassF1Score(num_classes=2, average=None)
    model.eval()
    with torch.no_grad():
        for step, (data, labels, _) in enumerate(test_loader):

            data = data.to(device)
            labels = labels.to(device)

            logits = model.forward(data)

            pred_labels = torch.argmax(logits, dim=1)

            accuracy_metric.update(pred_labels, labels)
            precision_metric.update(pred_labels, labels)
            recall_metric.update(pred_labels, labels)
            f1_metric.update(pred_labels, labels)

            if met_count < len(grad_cam):
                for i, (label, pred_label) in enumerate(zip(labels, pred_labels)):
                    pred_type = get_pred_type(label.item(), pred_label.item())
                    if len(grad_cam[pred_type]) < GRAD_CAM_COUNT:
                        grad_cam[pred_type].append([data[i], label])
                        if grad_cam[pred_type] == GRAD_CAM_COUNT:
                            met_count += 1

            del data, labels
    metrics["accuracy"] = accuracy_metric.compute().item()
    metrics["precision"] = precision_metric.compute()[0].item()
    metrics["recall"] = recall_metric.compute()[0].item()
    metrics["f1"] = f1_metric.compute()[0].item()

    metrics_path = os.path.join(save_folder, "metrics")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path) as f:
        json.dump(metrics, f)

    for pred_type, data in grad_cam.items():
        pred_type_path = os.path.join(save_folder, "gram-cam", pred_type)
        for image_label in data:
            image, label = image_label
            save_grad_cam(
                model,
                image,
                label,
                save_dir=pred_type_path,
            )
