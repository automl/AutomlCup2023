import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ConfigSpace import Constant

from .model_utils import get_eval_metric, get_model_save_path
from .model_utils import split_dataset, get_model
from .skorch_base import SkorchBaseModel


class Ensemble_Model():
    def __init__(self, metadata, input_cfg=None, seed=None):
        self.brute_force_ensemble = False
        self.num_greedy_iterations = 200
        self.metadata = metadata
        self.is_classification = (metadata.output_type.value == "classification")
        self.is_multi_label = (metadata.evaluation_metric.value in ['f1', 'bce'])
        self.run_statistics = {}
        self.seed = seed
        self.output_shape = tuple(metadata.output_shape)

        if input_cfg is None:
            input_cfg = [{'model_name': 'knn'},
                         {'model_name': 'xgb'},
                         {'model_name': 'autogluon'},
                         {'model_name': 'wrn'},
                         ]

        self.cfg = input_cfg
        self.executed_cfg = []
        self.model_list = []
        self.best_ensemble_weight = None
        self.begin_time = time.monotonic()
        self.end_time = self.begin_time + metadata.training_limit_sec
        self.eval_metric = get_eval_metric(metadata)

    def fit(self, X: np.ndarray, y: np.ndarray, X_unlabeled: np.ndarray = None, y_unlabeled: np.ndarray = None):
        torch.use_deterministic_algorithms(True)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        start_time = time.monotonic()
        self.X_train, self.X_val, self.y_train, self.y_val = split_dataset(X, y, self.metadata)

        checkpointed_model_list = []
        checkpointed_cfg = []

        for i, model_cfg in enumerate(self.cfg):
            sub_model_time_budget = (self.end_time - time.monotonic())
            if sub_model_time_budget < 600:
                continue
            sub_model_metadata = deepcopy(self.metadata)
            sub_model_metadata.training_limit_sec = int(sub_model_time_budget)
            sub_model_class = get_model(model_cfg['model_name'])

            sub_model = sub_model_class(sub_model_metadata, model_cfg, self.seed)
            try:
                if 'model_file' in model_cfg and model_cfg['model_file'] is not None:
                    # This case should not get called in the submission
                    model_path = Path(__file__).parent.parent.parent / 'results' / 'trained_models' / model_cfg[
                        'model_file']
                    assert model_path.exists(), f'Model file does not exist: {model_path}'
                    sub_model.load_model(str(model_path))
                    print(f'loaded parameters for {model_cfg["model_name"]}')
                else:
                    sub_model.fit(self.X_train, self.y_train, X_unlabeled, y_unlabeled, self.X_val, self.y_val)
                    if issubclass(sub_model_class, SkorchBaseModel) and (get_model_save_path() / sub_model.checkpoint_file).exists():
                        try:
                            checkpointed_model_cfg = deepcopy(model_cfg)
                            checkpointed_model_cfg["model_name"] = model_cfg["model_name"] + "_cp"
                            checkpointed_submodel = sub_model_class(sub_model_metadata, checkpointed_model_cfg, self.seed)
                            checkpointed_submodel.load_model(get_model_save_path() / sub_model.checkpoint_file)
                            checkpointed_model_list.append(checkpointed_submodel)
                            checkpointed_cfg.append(checkpointed_model_cfg)
                        except Exception as e:
                            print(e)
                self.model_list.append(sub_model)
                self.executed_cfg.append(sub_model.cfg)
                # only store statistics for fully trained model
                self.run_statistics[f"{model_cfg['model_name']}-model:{i}"] = sub_model.run_statistics
            except Exception as e:
                print(e)


        self.model_list.extend(checkpointed_model_list)
        self.executed_cfg.extend(checkpointed_cfg)

        model_val_pred = []
        for model in self.model_list:
            if self.is_classification:
                model_val_pred.append(model.predict_proba(self.X_val))
            else:
                model_val_pred.append(model.predict(self.X_val))
        possible_ensembles_weights = [tuple(map(int, format(i, f"0{len(self.model_list)}b"))) for i in
                                      range(1, 2 ** len(self.model_list))]
        if self.brute_force_ensemble:
            self.best_ensemble_weight, best_ensemble_error, best_ensemble_name = self.pick_best_ensemble(
                possible_ensembles_weights, model_val_pred)
        else:
            # https://arxiv.org/pdf/2307.08364.pdf Algorithm 1
            r = [0 for i in range(len(self.model_list))]
            H = []
            for i in range(self.num_greedy_iterations):
                R = [r[:i] + [r[i] + 1] + r[i + 1:] for i in range(len(self.model_list))]
                r, _, _ = self.pick_best_ensemble(R, model_val_pred)
                H += [r]
            self.best_ensemble_weight, best_ensemble_error, best_ensemble_name = self.pick_best_ensemble(H,
                                                                                                         model_val_pred)

        print("Best ensemble found was", best_ensemble_name, "with error", best_ensemble_error)

        self.run_statistics['fit_time'] = time.monotonic() - start_time

    def predict(self, X: np.ndarray):
        ensemble_test_pred = self.predict_helper(X)
        if self.is_classification:
            if self.metadata.evaluation_metric.value == "bce":
                ensemble_test_pred = ensemble_test_pred[:, 1, :]
            else:
                ensemble_test_pred = np.argmax(ensemble_test_pred, axis=1)
        return ensemble_test_pred

    def predict_proba(self, X: np.ndarray):
        if not self.is_classification:
            raise Exception("Can't use predict_proba in a regression task!")
        ensemble_test_pred_proba = self.predict_helper(X)
        return ensemble_test_pred_proba

    def predict_helper(self, X: np.ndarray):
        start_time = time.monotonic()
        if self.is_classification:
            ensemble_test_pred = np.sum(
                [self.model_list[i].predict_proba(X) * weight for i, weight in enumerate(self.best_ensemble_weight)],
                axis=0) / sum(self.best_ensemble_weight)
        else:
            ensemble_test_pred = np.sum(
                [self.model_list[i].predict(X) * weight for i, weight in enumerate(self.best_ensemble_weight)],
                axis=0) / sum(self.best_ensemble_weight)

        self.run_statistics['ensemble_time'] = time.monotonic() - start_time

        return ensemble_test_pred

    def pick_best_ensemble(self, possible_ensembles_weights, model_val_pred):
        best_ensemble_weight = None
        best_ensemble_error = float("+inf")
        best_ensemble_name = 0
        for ensemble_weight in possible_ensembles_weights:
            ensemble_val_pred = np.sum([model_val_pred[i] * weight for i, weight in enumerate(ensemble_weight)],
                                       axis=0) / sum(ensemble_weight)
            if self.is_classification:
                if self.metadata.evaluation_metric.value == "bce":
                    ensemble_val_pred = ensemble_val_pred[:, 1, :]
                else:
                    ensemble_val_pred = np.argmax(ensemble_val_pred, axis=1)
            ensemble_error = self.eval_metric(self.y_val, ensemble_val_pred)
            ensemble_name = [f"{weight}*{self.executed_cfg[i]['model_name']}-model:{i}" for i, weight in
                             enumerate(ensemble_weight)]
            #print(ensemble_name, ensemble_error)
            self.run_statistics['ensemble'] = self.run_statistics.get('ensemble', []) + [{
                'name': ', '.join(ensemble_name),
                'error': ensemble_error,
            }]
            if ensemble_error <= best_ensemble_error:
                best_ensemble_error = ensemble_error
                best_ensemble_weight = ensemble_weight
                best_ensemble_name = ensemble_name
        return best_ensemble_weight, best_ensemble_error, best_ensemble_name

    def save_model(self, save_path: str):
        """Saves model parameters to specified path. File should end with .pkl"""
        save_path = Path(save_path).parent
        timestamp = int(datetime.now().timestamp())
        for idx, model in enumerate(self.model_list):
            has_previous_model_save = 'model_file' in self.executed_cfg[idx] and self.executed_cfg[idx]['model_file'] is not None
            if hasattr(model, 'save_model') and not has_previous_model_save:
                model_file = f"{model.cfg['model_name']}_{timestamp}.pkl"
                model.save_model(str(save_path / model_file))
                self.executed_cfg[idx]['model_file'] = model_file
                timestamp += 1

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="ensemble")
        cs.add_hyperparameters([name])
        return cs
