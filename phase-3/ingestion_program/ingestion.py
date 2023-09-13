"""ingestion program for autoWSL"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import Timer, get_logger
from dataset import AutoMLCupDataset
from filelock import FileLock
from predict import predict

VERBOSITY_LEVEL = "INFO"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _cwd() -> Path:
    """Helper function for getting the current directory of this script."""
    return Path(__file__).resolve().parent


def write_start_file(output_dir: Path):
    """Create start file 'start.txt' in `output_dir` with updated timestamp
    start time.

    """
    LOGGER.info("===== alive_thd started")
    start_filepath = output_dir / "start.txt"
    lockfile = output_dir / "start.txt.lock"
    while True:
        current_time = datetime.datetime.now().timestamp()
        with FileLock(lockfile):
            with open(start_filepath, "w", encoding="utf-8") as ftmp:
                json.dump(current_time, ftmp)
        time.sleep(10)


class ModelApiError(Exception):
    """Model api error"""


class IngestionError(RuntimeError):
    """Model api error"""


def _parse_args():
    root_dir = _cwd()
    default_dataset_dir = root_dir / "sample_data"
    default_output_dir = root_dir / "sample_result_submission"
    default_ingestion_program_dir = root_dir / "ingestion_program"
    default_code_dir = root_dir / "code_submission"
    default_score_dir = root_dir / "scoring_output"
    default_temp_dir = root_dir / "temp_output"
    default_time_budget = 18000
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_dataset_dir,
        help="Directory storing the dataset metadata",
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default="/datasets",
        help="Directory storing the datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory storing the predictions. It will "
        "contain e.g. [start.txt, adult.predict_0, "
        "adult.predict_1, ..., end.txt] when ingestion "
        "terminates.",
    )
    parser.add_argument(
        "--ingestion_program_dir",
        type=Path,
        default=default_ingestion_program_dir,
        help="Directory storing the ingestion program "
        "`ingestion.py` and other necessary packages.",
    )
    parser.add_argument(
        "--code_dir",
        type=Path,
        default=default_code_dir,
        help="Directory storing the submission code "
        "`model.py` and other necessary packages.",
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=default_score_dir,
        help="Directory storing the scoring output "
        "e.g. `scores.txt` and `detailed_results.html`.",
    )
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=default_temp_dir,
        help="Directory storing the temporary output."
        "e.g. save the participants` model after "
        "trainning.",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=default_time_budget,
        help="Time budget for trainning model if not specified" " in meta.json.",
    )
    parser.add_argument(
        "--no-predict",
        dest="predict",
        default=True,
        action="store_false",
        help="Don't run test predictions.",
    )

    args = parser.parse_args()
    LOGGER.debug("Parsed args are: %s", args)
    LOGGER.debug("-" * 50)
    return args


def _init_python_path(args):
    sys.path.append(str(args.ingestion_program_dir))
    sys.path.append(str(args.code_dir))
    args.output_dir.mkdir(exist_ok=True)
    args.temp_dir.mkdir(exist_ok=True)


def _check_umodel_methed(umodel):
    # Check if the model has methods `train`, `predict`.
    for attr in ["train", "predict"]:
        if not hasattr(umodel, attr):
            raise ModelApiError(
                "Your model object doesn't have the method "
                f"`{attr}`. Please implement it in model.py."
            )


def _train(args, umodel, dataset: AutoMLCupDataset):
    # Train the model
    start = time.time()
    umodel.train(dataset.get_split("train"))
    duration = time.time() - start
    LOGGER.info(f"Finished training the model. time spent {duration:5.2} sec")

    result = {}
    result["duration"] = duration
    return result


def _predict(args, model):
    # Make predictions using the trained model
    LOGGER.info("===== call prediction")

    predict_args = argparse.Namespace(
        **{
            "dataset_dir": args.dataset_dir,
            "datasets_root": args.datasets_root,
            "model_dir": args.code_dir,
            "output_dir": args.output_dir,
            "temp_dir": args.temp_dir,
        }
    )
    result = predict(predict_args, model)
    return result


def _finalize(args, train_result, pred_result):
    if pred_result["status"] == "success":
        # Finishing ingestion program
        end_time = time.time()
        overall_time_spent = train_result["duration"] + pred_result["duration"]

        # Write overall_time_spent to a end.txt file
        end_filename = "end.txt"
        content = {"ingestion_duration": overall_time_spent, "end_time": end_time}

        with open(args.output_dir / end_filename, "w", encoding="utf-8") as ftmp:
            json.dump(content, ftmp)
            LOGGER.info(f"Wrote the file {end_filename} marking the end of ingestion.")

            LOGGER.info("[+] Done. Ingestion program successfully terminated.")
            LOGGER.info(f"[+] Overall time spent {overall_time_spent:5.2} sec")

        # Copy all files in output_dir to score_dir
        if args.output_dir != args.score_dir:
            os.system(f"cp -R {args.output_dir/ '*'} {args.score_dir}")
            LOGGER.debug("Copied all ingestion output to scoring output directory.")
        else:
            LOGGER.debug(
                "Skipped copying all ingestion output to scoring output directory."
            )

        LOGGER.info("[Ingestion terminated]")
    elif pred_result["status"] == "timeout":
        raise IngestionError("predicting timeout")
    else:
        raise IngestionError("error occurs when predicting")


def main():
    """main entry"""
    LOGGER.info("===== Start ingestion program.")
    # Parse directories from input arguments
    LOGGER.info("===== Initialize args.")
    args = _parse_args()
    _init_python_path(args)
    dataset = AutoMLCupDataset(args.dataset_dir, args.datasets_root)
    LOGGER.info(f"Time budget: {args.time_budget}")

    LOGGER.info("===== Set alive_thd")
    alive_thd = threading.Thread(
        target=write_start_file, name="alive", args=(args.output_dir,)
    )
    alive_thd.daemon = True
    alive_thd.start()

    LOGGER.info("===== Install user dependencies")
    requirements_txt = args.ingestion_program_dir / "requirements.txt"
    if requirements_txt.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "-r", requirements_txt])

    LOGGER.info("===== Load user model")
    # pylint: disable-next=import-error,import-outside-toplevel
    from model import Model

    umodel = Model(dataset.metadata())

    LOGGER.info("===== Check user model methods")
    _check_umodel_methed(umodel)

    timer = Timer()
    time_limit = dataset.metadata().training_limit_sec
    LOGGER.info(f"===== Setting time limit to {time_limit}")
    timer.set(time_limit)
    with timer.time_limit("train+predict"):
        LOGGER.info("===== Begin training user model")
        try:
            train_result = _train(args, umodel, dataset)
        except SystemExit as ex:
            LOGGER.exception(
                "Encountered system exit exception during training %s", str(ex)
            )
            raise ex
        except Exception as ex:
            LOGGER.exception("Encountered exception during training %s", str(ex))
            raise ex

        if not args.predict:
            LOGGER.info("===== Skipping prediction")
            return
        try:
            LOGGER.info("===== Begin preding by user model on test set")
            pred_result = _predict(args, umodel)
        except SystemExit as ex:
            LOGGER.exception(
                "Encountered system exit exception during training %s", str(ex)
            )
            raise ex
        except Exception as ex:
            LOGGER.exception("Encountered exception during prediction %s", str(ex))
            raise ex

    _finalize(args, train_result, pred_result)


if __name__ == "__main__":
    main()
