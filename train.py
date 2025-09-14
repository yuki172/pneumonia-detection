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
import mlflow

parser = argparse.ArgumentParser()

# training
parser.add_argument(
    "--epochs",
    type=int,
    dest="epochs",
    default=1,
    help="number of epochs to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    dest="batch_size",
    default=BATCH_SIZE,
    help="batch size",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    dest="learning_rate",
    default=LEARNING_RATE,
    help="learning rate",
)

# data
parser.add_argument(
    "--augment",
    type=int,
    dest="augment",
    help="augment dataset",
)
parser.add_argument(
    "--max_samples",
    type=int,
    dest="max_samples",
    help="max number of samples to take in datasets",
)

# model saving
parser.add_argument(
    "--save_epochs",
    type=int,
    dest="save_epochs",
    default=1,
    help="save at this number of epochs ",
)
parser.add_argument(
    "--save_base_folder",
    type=str,
    dest="save_base_folder",
    default="trained/",
    help="base directory for saving models params and training results",
)
parser.add_argument(
    "--pretrained_path",
    type=str,
    dest="pretrained_path",
    help="pretrained path for model and optimizer",
)

parser.add_argument(
    "--on_kaggle",
    type=bool,
    dest="on_kaggle",
    help="running on kaggle",
)


# eval
parser.add_argument(
    "--eval_epochs",
    type=int,
    dest="eval_epochs",
    default=1,
    help="evaluate at this number of epochs",
)


args = parser.parse_args()


def train(
    epochs=1,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    save_epochs=1,
    eval_epochs=1,
    save_base_folder="trained/",
    max_samples=None,
    augment=True,
    pretrained_path=None,
):
    for name, arg in [
        ["epochs", epochs],
        ["batch_size", batch_size],
        ["learning_rate", learning_rate],
        ["save_epochs", save_epochs],
        ["eval_epochs", eval_epochs],
        ["save_base_folder", save_base_folder],
        ["max_samples", max_samples],
        ["pretrained_path", pretrained_path],
    ]:
        print(name, arg)

    model = DetectionModel()
    train_set = PneumoniaDetectionDataset(
        split="train", augment=augment, normalize=True
    )
    eval_set = PneumoniaDetectionDataset(split="val", augment=False, normalize=True)
    loss_fn = nn.CrossEntropyLoss()

    if max_samples != None:
        train_set = torch.utils.data.Subset(train_set, range(max_samples))
        eval_set = torch.utils.data.Subset(eval_set, range(max_samples))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device", device)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    if pretrained_path:
        checkpoint = torch.load(
            f"{pretrained_path}checkpoint.pth", map_location=torch.device("cpu")
        )
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        model = torch.load(f"{pretrained_path}model.pth", weights_only=False)

    train_num_batches = len(train_loader)
    eval_num_batches = len(eval_loader)

    print("train num batches", train_num_batches)
    print("eval num batches", eval_num_batches)

    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "eval_precision": [],
        "eval_recall": [],
        "eval_f1": [],
        "best_eval_loss": None,
    }

    metrics_path = f"{save_base_folder}metrics.json"

    def log_metric(metric, value, epoch):
        metrics[metric].append([epoch, value])
        mlflow.log_metric(metric, value, step=epoch)

    try:
        with open(metrics_path, "r") as file:
            metrics = json.load(file)
    except:
        print("metrics json not found")

    print("\nStart training...\n")
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss = 0
        train_accuracy_metric = MulticlassAccuracy(num_classes=2, average="micro")

        progress_bar = tqdm(range(train_num_batches), position=0, leave=False)

        model.train()

        with mlflow.start_run():
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lr", learning_rate)
            mlflow.log_param("model", "resnet50")
            mlflow.log_param("augment", augment)
            for step, (data, labels, _) in enumerate(train_loader):

                data = data.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model.forward(data)
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

                progress_bar.update(1)

                train_loss += loss.item() / train_num_batches
                pred_labels = torch.argmax(logits, dim=1)
                train_accuracy_metric.update(pred_labels, labels)

                del data, labels

            log_metric("train_loss", train_loss, epoch)
            log_metric("train_accuracy", train_accuracy_metric.compute().item(), epoch)

            if (epoch + 1) % save_epochs == 0:

                eval_loss = 0
                accuracy_metric = MulticlassAccuracy(num_classes=2, average="micro")
                precision_metric = MulticlassPrecision(num_classes=2, average=None)
                recall_metric = MulticlassRecall(num_classes=2, average=None)
                f1_metric = MulticlassF1Score(num_classes=2, average=None)
                model.eval()
                with torch.no_grad():
                    for step, (data, labels, _) in enumerate(eval_loader):

                        data = data.to(device)
                        labels = labels.to(device)

                        logits = model.forward(data)
                        loss = loss_fn(logits, labels)

                        eval_loss += loss.item() / eval_num_batches

                        pred_labels = torch.argmax(logits, dim=1)

                        # print("---")
                        # print(logits)
                        # print("pred_labels", list(pred_labels.numpy()))
                        # print("labels", list(labels.numpy()))

                        accuracy_metric.update(pred_labels, labels)
                        precision_metric.update(pred_labels, labels)
                        recall_metric.update(pred_labels, labels)
                        f1_metric.update(pred_labels, labels)

                        del data, labels

                log_metric("eval_loss", eval_loss, epoch)
                log_metric("eval_accuracy", accuracy_metric.compute().item(), epoch)
                log_metric(
                    "eval_precision", precision_metric.compute()[0].item(), epoch
                )
                log_metric("eval_recall", recall_metric.compute()[0].item(), epoch)

                os.makedirs(os.path.dirname(save_base_folder), exist_ok=True)

                with open(f"{save_base_folder}metrics.json", "w") as f:
                    json.dump(metrics, f)

                if (
                    not metrics["best_eval_loss"]
                    or metrics["best_eval_loss"][1] > eval_loss
                ):
                    if metrics["best_eval_loss"]:
                        prev_epoch = metrics["best_eval_loss"][0]
                        prev_best_path = f"{save_base_folder}epoch_{prev_epoch}"
                        if os.path.exists(prev_best_path):
                            shutil.rmtree(prev_best_path)

                    metrics["best_eval_loss"] = [epoch, eval_loss]

                    curr_best_path = f"{save_base_folder}epoch_{epoch}/"
                    os.makedirs(os.path.dirname(curr_best_path), exist_ok=True)

                    torch.save(model, f"{curr_best_path}/model.pth")
                    checkpoint = {
                        "epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, f"{curr_best_path}/checkpoint.pth")

    print("\nEnd training\n")


train(
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    save_epochs=args.save_epochs,
    eval_epochs=args.eval_epochs,
    save_base_folder=args.save_base_folder,
    augment=args.augment,
    max_samples=args.max_samples,
    pretrained_path=args.pretrained_path,
)
