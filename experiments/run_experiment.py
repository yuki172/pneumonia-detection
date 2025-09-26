import argparse
import yaml
import mlflow
from mlflow.pytorch import log_model
from train import train
from utils import get_run_name
import numpy as np


def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_name = get_run_name(**config)
    mlflow.set_experiment("pneumonia_detection")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        best_model, metrics = train(**config)

        for epoch in range(config["epochs"]):
            for key, metric in metrics.items():
                if key != "best_eval_loss":
                    mlflow.log_metric(key, metric[epoch][1], step=epoch)

        # # log model
        # input_example = np.random.default_rng().random(
        #     [1, 3, 224, 224], dtype="float32"
        # )
        # log_model(best_model, "model", input_example=input_example)


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
args = parser.parse_args()
if args.config_path:
    run_experiment(args.config_path)
