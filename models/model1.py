import os
import time
from pathlib import Path
import json
import glob

import numpy as np
from datasets import Dataset

from base_models.model_utils import instantiate_model, get_model
from p1metadata import *

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Model:
    def __init__(self, time_budget=None, seed=None):
        self.sub_model = None
        self.input_shape = None
        self.output_shape = None
        self.num_classes = None
        self.max_len = None
        self.time_budget = time_budget
        self.seed = seed

    def train(self, train_dataset: Dataset):
        print(f"FROM MODEL.PY: TRAIN {train_dataset}")

        max_len = max(map(len, train_dataset["input"]))
        self.max_len = max_len
        X = np.zeros((len(train_dataset), max_len), dtype=np.float32)
        for idx, row in enumerate(train_dataset["input"]):
            X[idx, : len(row)] = row

        y = np.array(train_dataset["label"])

        print(f"FROM MODEL.PY: X SHAPE {X.shape}")
        print(f"FROM MODEL.PY: Y SHAPE {y.shape}")

        self.input_shape = X.shape[1:]
        self.output_shape = y.shape[1:]
        if self.time_budget is None:
            time_budget = 18000
        else:
            time_budget = self.time_budget
        time_budget *= 0.95

        begin_time = time.monotonic()
        end_time = begin_time + time_budget

        self.metadata = AutoMLCupMetadata(1, int(np.max(X) + 1), InputShape(X.shape[0], X.shape[1], 1, 1, 1),
                                     [np.max(y) + 1],
                                     OutputType("classification"), EvaluationMetric("accuracy"), int(time_budget))
        X = X.reshape((X.shape[0], X.shape[1], 1, 1, 1))
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
        self.sub_model.fit(X, y)

    def predict(self, dataset: Dataset):
        print(f"FROM MODEL.PY: TEST {dataset}")

        print('Training and predict same length: ', self.max_len == max(map(len, dataset["input"])))
        X = np.zeros((len(dataset), self.max_len), dtype=np.float32)
        for idx, row in enumerate(dataset["input"]):
            X[idx, : len(row)] = row

        X = X.reshape((X.shape[0], X.shape[1], 1, 1, 1))
        y = self.sub_model.predict(X)
        return y
