import argparse
import yaml
import mlflow
from mlflow.pytorch import log_model
from ..train import train
from ..utils import get_run_name


def run(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_name = get_run_name(**config)
    mlflow.set_experiment("pneumonia_detection")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        best_model, metrics = train(**config)

        for epoch in range(config.epochs):
            for key, metric in metrics.items():
                if key != "best_eval_loss":
                    mlflow.log_metric(key, metric[epoch], step=epoch)

        # log model
        log_model(best_model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    run(args.config_path)
