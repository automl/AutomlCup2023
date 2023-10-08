import random
import time
from abc import ABC
from timeit import default_timer as timer

import numpy as np
import torch
from ConfigSpace import EqualsCondition, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import Dataset
from torchvision.transforms import TrivialAugmentWide, RandAugment

from .model_utils import is_multi_label


class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        assert len(x) == len(y)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform and x.shape[-3] in [1, 3] and x.shape[-2] > 1 and x.shape[-1] > 1 and x.shape[-4] == 1:
            x = (x * 255).to(torch.uint8)
            x = self.transform(x)
            x = x.to(torch.float32) / 255
        return x, y


class BaseModel(ABC):

    def __init__(self, metadata, input_cfg=None, seed=None):
        self.model = None
        self.metadata = metadata
        self.input_shape = (
            metadata.input_shape.max_sequence_len, metadata.input_shape.channels, metadata.input_shape.width,
            metadata.input_shape.height)
        self.output_shape = tuple(metadata.output_shape)
        self.time_budget = metadata.training_limit_sec
        self.is_classification = (metadata.output_type.value == "classification")

        self.is_multi_label = is_multi_label(metadata)
        self.run_statistics = {}
        self.seed = seed
        self.cfg = dict(self.get_config_space().get_default_configuration())
        if input_cfg is None:
            input_cfg = {}
        self.cfg.update(input_cfg)
        torch.use_deterministic_algorithms(True)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_unlabeled: np.ndarray = None, y_unlabeled: np.ndarray = None,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        torch.use_deterministic_algorithms(True)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        batch_size_picker = True
        print("Start Training", self.__class__.__name__)
        start = timer()

        augmentation = None
        if self.cfg['augmentation'] == 'TrivialAugment':
            augmentation = TrivialAugmentWide()
        elif self.cfg['augmentation'] == 'RandAugment':
            augmentation = RandAugment(num_ops=self.cfg['num_ops'], magnitude=self.cfg['magnitude'])

        batch_size = self.cfg['starting_batch_size']
        self.model.set_params(max_epochs=1)
        i = 0
        while i < self.cfg['max_epochs']:
            if time.monotonic() > self.begin_time + self.time_budget:
                break
            if batch_size_picker:
                while True:
                    self.model.set_params(batch_size=batch_size)
                    try:
                        self.model.partial_fit(CustomDataset(X, y, transform=augmentation), y)
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        print(e)
                        torch.cuda.empty_cache()
                        batch_size //= 2
                        print("Decreased batch size to", batch_size)
            else:
                self.model.partial_fit(CustomDataset(X, y, transform=augmentation), y)
            i += 1
            if time.monotonic() > self.begin_time + self.time_budget:
                break
            if (not self.cfg['use_ssl']) \
                    or (X_unlabeled is None or len(X_unlabeled) == 0) \
                    or i < self.cfg['fixmatch_warmup']:
                continue
            elif self.is_classification:
                # SSL for classification
                # https://github.com/skorch-dev/skorch/issues/805
                prob_distributions = self.predict_proba(X_unlabeled)
                labels = np.argmax(prob_distributions, axis=1)
                highest_prob = np.squeeze(np.max(prob_distributions, axis=1), axis=1)
                newly_labeled = highest_prob > self.cfg['fixmatch_threshold_classification']
                if self.is_multi_label or not self.is_classification:
                    labels = labels.astype("float32")
                X_unlabeled, X_newly_labeled = X_unlabeled[~newly_labeled], X_unlabeled[newly_labeled]
                X = np.concatenate((X, X_newly_labeled))
                y = np.concatenate((y, labels[newly_labeled]))
            elif hasattr(self.model.module_, 'embedding'):
                n_neighbors = self.cfg['n_neighbors']
                self.model.module_.embedding = True
                embedding_labeled = np.array(self.model.forward(X))
                embedding_unlabeled = np.array(self.model.forward(X_unlabeled))
                self.model.module_.embedding = False
                knn = KNeighborsRegressor(n_neighbors=n_neighbors).fit(embedding_labeled, y)
                neighbors_indices = []
                step_size = 10
                for step in range(0, len(embedding_unlabeled), step_size):
                    slice = embedding_unlabeled[step:step + step_size]
                    if len(slice.shape) == 1:
                        slice = slice.reshape(1, -1)
                    neighbors_indices.append(knn.kneighbors(slice, return_distance=False))
                neighbors_indices = np.concatenate(neighbors_indices, axis=0)
                prediction_neighbors = y[neighbors_indices]
                labels = np.mean(prediction_neighbors, axis=1)
                newly_labeled = (
                            np.std(prediction_neighbors, axis=1) < self.cfg['fixmatch_threshold_regression']).flatten()
                X_unlabeled, X_newly_labeled = X_unlabeled[~newly_labeled], X_unlabeled[newly_labeled]
                X = np.concatenate((X, X_newly_labeled))
                y = np.concatenate((y, labels[newly_labeled]))

        end = timer()
        self.run_statistics['runtime'] = end - start
        print("Stopped Training", self.__class__.__name__, f"took {self.run_statistics['runtime']:.1f}")

    def predict(self, X: np.ndarray):
        y = self.model.predict(X)
        return y

    def predict_proba(self, X: np.ndarray):
        if not self.is_classification:
            raise Exception("Can't use predict_proba in a regression task!")
        return self.model.predict_proba(X)

    @classmethod
    def get_config_space(cls):
        cs = ConfigurationSpace()
        fixmatch_warmup = UniformIntegerHyperparameter(
            name="fixmatch_warmup", lower=0, upper=50, default_value=30, log=False
        )
        fixmatch_threshold_classification = UniformFloatHyperparameter(
            name="fixmatch_threshold_classification", lower=0.6, upper=0.975, default_value=0.9, log=False
        )
        fixmatch_threshold_regression = UniformFloatHyperparameter(
            name="fixmatch_threshold_regression", lower=0.01, upper=0.4, default_value=0.25, log=False
        )
        starting_batch_size = CategoricalHyperparameter(
            name="starting_batch_size", choices=[256, 512], default_value=512
        )
        use_ssl = CategoricalHyperparameter(
            name="use_ssl", choices=[True, False], default_value=False
        )
        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=3, upper=20, default_value=5, log=False
        )
        augmentation = CategoricalHyperparameter(
            name="augmentation", choices=['RandAugment', 'TrivialAugment', 'None'], default_value='None'
        )
        magnitude = UniformIntegerHyperparameter(
            name="magnitude", lower=1, upper=20, default_value=9, log=False
        )
        num_ops = UniformIntegerHyperparameter(
            name="num_ops", lower=1, upper=5, default_value=2, log=False
        )
        cs.add_hyperparameters(
            [fixmatch_threshold_regression, fixmatch_threshold_classification, fixmatch_warmup, starting_batch_size,
             use_ssl, magnitude, augmentation, n_neighbors, num_ops])
        cs.add_conditions([EqualsCondition(magnitude, augmentation, 'RandAugment'),
                           EqualsCondition(num_ops, augmentation, 'RandAugment'),
                           EqualsCondition(fixmatch_warmup, use_ssl, True),
                           EqualsCondition(fixmatch_threshold_regression, use_ssl, True),
                           EqualsCondition(fixmatch_threshold_classification, use_ssl, True),
                           EqualsCondition(n_neighbors, use_ssl, True)])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return False
