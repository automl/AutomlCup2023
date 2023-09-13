from pathlib import Path

import numpy as np

from dataloader import AutoMLCupDataloader


class Camelyon17Dataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "camelyon17"

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)

        self.train = None
        self.test = None

    def get_split(self, split):
        if split == "train":
            if self.train is None:
                x_train = np.load(self.directory / "x_train.npz")["x_train"]
                y_train = np.load(self.directory / "y_train.npz")["y_train"]
                self.train = {
                    "input": x_train,
                    "label": y_train,
                }
            return self.train
        elif split == "test":
            if self.test is None:
                x_test = np.load(self.directory / "x_test.npz")["x_test"]
                y_test = np.load(self.directory / "y_test.npz")["y_test"]
                self.test = {
                    "input": x_test,
                    "label": y_test,
                }
            return self.test
