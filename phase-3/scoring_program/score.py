import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.nn.functional import binary_cross_entropy, l1_loss, mse_loss
import torch

from common import get_logger
from dataset import AutoMLCupDataset
from metadata import EvaluationMetric


VERBOSITY_LEVEL = "INFO"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def parse_args():
    default_app_dir = Path("/app/")
    default_input_dir = default_app_dir / "input"
    default_prediction_dir = default_input_dir / "res"
    default_data_dir = default_input_dir / "ref"
    default_output_data_dir = default_app_dir / "output"

    parser = ArgumentParser()
    parser.add_argument("--prediction_dir", default=default_prediction_dir, type=Path)
    parser.add_argument("--dataset_dir", default=default_data_dir, type=Path)
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default="/datasets",
        help="Directory storing the datasets",
    )
    parser.add_argument("--output_dir", default=default_output_data_dir, type=Path)

    args = vars(parser.parse_args())
    return args


def write_scores(score_file: Path, scores: dict):
    with open(score_file, "w", encoding="utf-8") as score_file_obj:
        json.dump(scores, score_file_obj)


def get_duration(prediction_metadata_file):
    with open(
        prediction_metadata_file, "r", encoding="utf-8"
    ) as prediction_metadata_file_obj:
        metadata = json.load(prediction_metadata_file_obj)
        return metadata["ingestion_duration"]


def calculate_error(
    y_test: ArrayLike, y_pred: ArrayLike, evaluation_metric: EvaluationMetric
) -> float:
    LOGGER.info(f"===== Scoring with '{evaluation_metric.value}'.")

    if evaluation_metric is EvaluationMetric.ACCURACY:
        return 1 - accuracy_score(y_test, y_pred)
    if evaluation_metric is EvaluationMetric.BCE:
        return binary_cross_entropy(
            Tensor(y_pred), Tensor(y_test), reduction="mean"
        ).item()
    if evaluation_metric is EvaluationMetric.MAE:
        return l1_loss(Tensor(y_pred), Tensor(y_test), reduction="mean").item()
    if evaluation_metric is EvaluationMetric.MSE:
        return mse_loss(Tensor(y_pred), Tensor(y_test), reduction="mean").item()
    if evaluation_metric is EvaluationMetric.RMSE:
        return torch.sqrt(
            mse_loss(Tensor(y_pred), Tensor(y_test), reduction="mean")
        ).item()
    if evaluation_metric is EvaluationMetric.F1:
        return 1 - f1_score(y_test, y_pred)
    raise ValueError(f"EvaluationMetric '{evaluation_metric}' is invalid.")


def main():
    LOGGER.info("===== Start scoring program.")
    LOGGER.info("===== Initialize args.")
    args = parse_args()

    output_dir: Path = args["output_dir"]
    output_dir.mkdir(exist_ok=True)

    prediction_file = args["prediction_dir"] / "prediction.npz"
    score_file = output_dir / "scores.json"

    LOGGER.info("===== Reading dataset.")
    dataset = AutoMLCupDataset(args["dataset_dir"], args["datasets_root"])

    LOGGER.info("===== Getting test labels.")
    y_test = dataset.get_split("test")["label"]

    LOGGER.info("===== Loading predictions.")
    y_pred = np.load(prediction_file)["prediction"]

    LOGGER.info("===== Getting scores.")
    error_value = calculate_error(y_test, y_pred, dataset.metadata().evaluation_metric)
    LOGGER.info(f"===== Got error value of '{error_value}'.")

    LOGGER.info("===== Getting runtime duration.")
    duration = get_duration(args["prediction_dir"] / "end.txt")
    LOGGER.info(f"===== Got runtime duration of '{duration}'.")

    scores = {
        "error": error_value,
        "duration": duration,
    }
    LOGGER.info(f"===== Writing scores to '{score_file}'.")
    write_scores(score_file, scores)


if __name__ == "__main__":
    main()
