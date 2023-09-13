"""User model."""

import math
import os
from pathlib import Path
import json
import glob
from typing import Dict

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler

from base_models.model_utils import instantiate_model, is_multi_label, get_model
from preprocessing import InputTransformer

from copy import deepcopy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Model:
    """User model."""

    def __init__(self, metadata):
        """Initialize the model."""
        self.metadata = deepcopy(metadata)
        print(f"FROM MODEL.PY: METADATA {metadata}")
        print(f"INPUT_DIMENSION {metadata.input_dimension}")
        print(f"NUM_EXAMPLES {metadata.input_shape.num_examples}")
        print(f"MAX_SEQUENCE_LEN {metadata.input_shape.max_sequence_len}")
        print(f"CHANNELS {metadata.input_shape.channels}")
        print(f"WIDTH {metadata.input_shape.width}")
        print(f"HEIGHT {metadata.input_shape.height}")
        print(f"OUTPUT_SHAPE {metadata.output_shape}")
        print(f"OUTPUT_TYPE {metadata.output_type}")
        print(f"EVALUATION_METRIC {metadata.evaluation_metric}")
        self.input_transformer = InputTransformer(self.metadata)
        self.output_transformer = StandardScaler()  # MinMaxScaler((-1, 1))
        self.sub_model = None
        self.metadata.training_limit_sec = int(self.metadata.training_limit_sec * 0.95)

    def train(self, train_dataset: Dict[str, ArrayLike]):
        """Trains the model.

        Args:
            train_dataset (Dict[str, ArrayLike]): The training dataset.
        """
        print("FROM MODEL.PY: TRAIN")
        print(f"FROM MODEL.PY: CUDA {torch.cuda.is_available()}")
        print(f"FROM MODEL.PY: SELF {self}")

        try:
            train_input = np.array(train_dataset["input"])
            train_dataset["input"] = None
        except ValueError:
            max_len = max(map(len, train_dataset["input"]))
            train_input = np.zeros((len(train_dataset), max_len), dtype=np.float32)
            for idx, row in enumerate(train_dataset["input"]):
                print(idx)
                train_input[idx, : len(row)] = row

        train_label = np.array(train_dataset["label"])
        train_dataset["label"] = None
        X = train_input
        y = train_label
        del train_input
        del train_label

        self.metadata = self.input_transformer.fit(X)
        X = self.input_transformer.transform(X)

        if self.metadata.output_type.value == "regression":
            shape = y.shape
            y = y.reshape(len(y), -1)
            y = self.output_transformer.fit_transform(y)
            y = y.reshape(shape)

        if len(y.shape) == 1:
            y = y.reshape(y.shape + (1,))
        labeled_datapoints = ~np.isnan(y).any(axis=tuple(range(1, y.ndim)))
        X_labeled, X_unlabeled = X[labeled_datapoints], X[~labeled_datapoints]
        y_labeled, y_unlabeled = y[labeled_datapoints], y[~labeled_datapoints]
        del X
        del y
        print("Loaded Dataset")

        # remove one-hot encoding of labels
        if self.metadata.output_type.value == "classification":
            if not is_multi_label(self.metadata):
                y_labeled = np.argmax(y_labeled, axis=1)

        models = ["mlp", "efn", "multires", "seq_transformer", "vit", "wrn", "unet"]
        models = [e for e in models if get_model(e).is_applicable(self.metadata)]
        print(models)
        model_configs = []
        for e in models:
            config_list = sorted(glob.glob(str(Path(__file__).parent / "configs" / (e +"_*.json"))))
            # if len(config_list) == 0:
            #     print("no configs found for", e, "loading default config")
            #     model_configs.append({"model_name": e})
            for cfg_file in config_list:
                print("loading", cfg_file)
                fp=open(cfg_file, 'r')
                model_configs.append(json.load(fp))
                fp.close()
        self.sub_model = instantiate_model("ensemble", self.metadata, model_configs)
        self.sub_model.fit(X_labeled, y_labeled, X_unlabeled, y_unlabeled)

        print(f"FROM MODEL.PY: INPUT SHAPE {X_labeled.shape}")
        print(f"FROM MODEL.PY: LABEL SHAPE {y_labeled.shape}")

    def predict(self, prediction_dataset: Dict[str, ArrayLike]) -> ArrayLike:
        """Predicts over a prediction dataset using the model.

        Args:
            prediction_dataset (Dict[str, ArrayLike]): Dataset to use for prediction.

        Returns:
            ArrayLike: The predictions.
        """
        print("FROM MODEL.PY: PREDICT")
        print(f"FROM MODEL.PY: SELF {self}")

        test_input = prediction_dataset["input"]
        test_input = self.input_transformer.transform(test_input)
        pred = self.sub_model.predict(test_input)
        if self.sub_model.is_classification:
            if not self.sub_model.is_multi_label:
                pred = np.identity(math.prod(self.sub_model.output_shape))[pred]
        if self.metadata.output_type.value == "regression":
            shape = pred.shape
            pred = pred.reshape(len(pred), -1)
            pred = self.output_transformer.inverse_transform(pred)
            pred = pred.reshape(shape)

        if self.metadata.output_type.value == "regression" and len(pred.shape) == 2 and pred.shape[1] == 1:
            pred = np.squeeze(pred, axis=1)
        return pred
