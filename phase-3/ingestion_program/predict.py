"""prediction"""
import argparse
import json
import os
from os.path import join
from pathlib import Path
from sys import path
from time import time

import numpy as np
from numpy.typing import ArrayLike

from common import get_logger
from dataset import AutoMLCupDataset
from pandas import Series

VERBOSITY_LEVEL = "WARNING"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _write_predict(output_dir, prediction: ArrayLike):
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.debug(f"prediction shape: {prediction.shape}")
    np.savez_compressed(join(output_dir, "prediction"), prediction=prediction)


def predict(args, umodel):
    """prediction"""
    result = {}
    try:
        dataset = AutoMLCupDataset(args.dataset_dir, args.datasets_root)
        path.append(args.model_dir)
        LOGGER.info("==== Load user model")

        LOGGER.info("==== start predicting")
        start = time()
        y_pred = umodel.predict({k: dataset.get_split("test")[k] for k in ["input"]})
        duration = time() - start
        LOGGER.info(f"Finished predicting the model. time spent {duration:5.2} sec")
        # Write predictions to output_dir
        _write_predict(args.output_dir, y_pred)
        result["status"] = "success"
        result["duration"] = duration
    except Exception as ex:
        LOGGER.error(ex, exc_info=True)
        result["status"] = "failed"

    return result


def _write_result(args, result):
    with open(args.result_file, "w", encoding="utf-8") as ftmp:
        json.dump(result, ftmp)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory storing the dataset (containing " "e.g. adult.data/)",
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default="/datasets",
        help="Directory storing the datasets",
    )
    parser.add_argument("--model_dir", type=str, help="Directory storing the model ")

    parser.add_argument("--result_file", type=str, help="a json file save the result")

    parser.add_argument(
        "--temp_dir",
        type=str,
        help="Directory storing the temporary output."
        "e.g. save the participants` model "
        "after trainning.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory storing the predictions. It will "
        "contain e.g. [start.txt, adult.predict_0, "
        "adult.predict_1, ..., end.txt] when ingestion "
        "terminates.",
    )

    parser.add_argument(
        "--pred_time_budget",
        type=float,
        help="Time budget for predicting model " "if not specified in meta.json.",
    )

    args = parser.parse_args()
    return args


def main():
    """main entry"""
    LOGGER.info("==== prediction process started")
    args = _parse_args()
    # pylint: disable-next=import-error,import-outside-toplevel
    from model import Model

    umodel = Model()
    result = predict(args, umodel)
    LOGGER.info("==== Write result")
    _write_result(args, result)


if __name__ == "__main__":
    main()
