import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from common import get_logger
from dataset import AutoMLCupDataset
from sklearn.metrics import accuracy_score

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
    parser.add_argument("--output_dir", default=default_output_data_dir, type=Path)

    args = vars(parser.parse_args())
    return args


def read_scores(score_file: Path) -> dict:
    if not score_file.exists():
        return {}
    with open(score_file, "r", encoding="utf-8") as score_file_obj:
        scores = json.load(score_file_obj)
        return scores


def write_scores(score_file: Path, scores: dict):
    with open(score_file, "w", encoding="utf-8") as score_file_obj:
        json.dump(scores, score_file_obj)


def get_duration(prediction_metadata_file):
    with open(
        prediction_metadata_file, "r", encoding="utf-8"
    ) as prediction_metadata_file_obj:
        metadata = json.load(prediction_metadata_file_obj)
        return metadata["ingestion_duration"]


def main():
    args = parse_args()

    output_dir: Path = args["output_dir"]
    output_dir.mkdir(exist_ok=True)

    prediction_file = args["prediction_dir"] / "prediction"
    score_file = output_dir / "scores.json"

    dataset = AutoMLCupDataset(args["dataset_dir"])
    y_test = np.array(dataset.get_split("test")["label"])

    y_pred = np.genfromtxt(prediction_file, skip_header=1)

    accuracy = accuracy_score(y_test, y_pred)

    scores = {
        "accuracy": accuracy,
        "duration": get_duration(args["prediction_dir"] / "end.txt"),
    }
    write_scores(score_file, scores)


if __name__ == "__main__":
    main()
