import time
from datetime import datetime

import numpy as np
import torch
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import Callback, Checkpoint
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from .base_model import BaseModel
from .model_utils import get_loss_function, split_dataset, get_model_save_path


class TimeOut(Callback):
    def __init__(self, begin_time, time_budget):
        self.time_budget = time_budget
        self.begin_time = begin_time

    def on_batch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        if time.monotonic() > self.begin_time + self.time_budget:
            print(
                f"Time budget of {self.time_budget:.1f}s exceeded by {time.monotonic() - (self.begin_time + self.time_budget):.1f}s")
            raise KeyboardInterrupt


def skorch_split(metadata):
    def skorch_internal_split(X, y):
        X_train, X_test, y_train, y_test = split_dataset(X.X, y, metadata)
        dataset_train = Dataset(X_train, y_train)
        dataset_valid = Dataset(X_test, y_test)
        return dataset_train, dataset_valid

    return skorch_internal_split


class SkorchBaseModel(BaseModel):
    def __init__(self, metadata, input_cfg, seed=None):
        super().__init__(metadata, input_cfg, seed)
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.dtype_input = None
        # use callbacks to inject early stopping, scoring or schedulers
        self.callbacks = []
        # period = self.cfg["period"]
        # self.callbacks.append(EarlyStopping(monitor="valid_loss", patience=period))  # , patience=10
        # self.callbacks.append(LRScheduler(WarmRestartLR, min_lr=2e-5, max_lr=1e-2, base_period=period, period_mult=1))
        self.begin_time = time.monotonic()
        self.callbacks.append(TimeOut(self.begin_time, self.time_budget))  # , patience=10
        self.checkpoint_file = f"{self.cfg['model_name']}_{int(datetime.now().timestamp())}.pt"
        self.callbacks.append(Checkpoint(dirname=get_model_save_path(), f_params=self.checkpoint_file))
        self.criterion = get_loss_function(metadata)
        # callbacks.append(EpochScoring(scoring="accuracy", name="train_accuracy", on_train=True))
        if self.is_classification:
            self.model_class = NeuralNetClassifier
        else:
            self.model_class = NeuralNetRegressor

        self.skorch_kwargs = {
            'device': self.device,
            'iterator_train__shuffle': True,
            'criterion': self.criterion
        }

    def fit(self, X: np.ndarray, y: np.ndarray, X_unlabeled: np.ndarray = None, y_unlabeled: np.ndarray = None,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        X = X.reshape((-1,) + self.input_shape)
        X_val = X_val.reshape((-1,) + self.input_shape)
        X = X.astype(
            "float32" if self.dtype_input is None else self.dtype_input)  # Todo: some models want int. Maybe save the datatype needed.
        X_val = X_val.astype(
            "float32" if self.dtype_input is None else self.dtype_input)  # Todo: some models want int. Maybe save the datatype needed.
        if self.is_multi_label or not self.is_classification:
            y = y.astype("float32")
            y_val = y_val.astype("float32")

        self.model.set_params(train_split=predefined_split(Dataset(X_val, y_val)))
        super().fit(X, y, X_unlabeled, y_unlabeled, X_val, y_val)

    def predict(self, X: np.ndarray):
        X = X.reshape((-1,) + self.input_shape)
        X = X.astype("float32" if self.dtype_input is None else self.dtype_input)
        y = super().predict(X)
        return y

    def predict_proba(self, X: np.ndarray):
        X = X.reshape((-1,) + self.input_shape)
        X = X.astype("float32" if self.dtype_input is None else self.dtype_input)
        y = super().predict_proba(X)
        return y

    def save_model(self, save_path: str):
        """Saves model parameters to specified path. File should end with .pkl"""
        self.model.save_params(f_params=save_path)  # additionally history and optimize can be saved

    def load_model(self, load_path):
        """Loads and initalizes model with parameters of saved model specified by path."""
        self.model.initialize()
        self.model.load_params(f_params=load_path)

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        return cs
