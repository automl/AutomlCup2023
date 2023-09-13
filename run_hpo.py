# ToDo: maybe a time budget for a single model?

import argparse
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from ConfigSpace import Configuration
from smac import Scenario, HyperparameterOptimizationFacade, MultiFidelityFacade

from models.base_models.base_model import BaseModel
from models.base_models.model_utils import get_model, get_model_save_path, is_multi_label
from run_config import score_prediction, get_formatted_dataset


def get_train_function(model_cls: BaseModel, suite: str, dataset_id: str, dataset_dir: str, time_budget: int, cv=1) -> \
        Callable[[Configuration, int], float]:
    """
    Returns a training function for the smac optimzer.
    This function inserts data loading and time management into the returned smac training function.
    """

    begin_time = time.monotonic()
    end_time = begin_time + time_budget

    def train_function(config: Configuration, seed: int, budget: float = None) -> float:
        """ Training function for smac optimizer. """
        total_score = 0.
        for fold in range(cv):

            metadata, output_transformer, X_train, y_train, X_test, y_test, X_unlabeled, y_unlabeled, X_val, y_val = \
                get_formatted_dataset(suite, dataset_id, dataset_dir, fold=fold, cv=cv)
            config = deepcopy(dict(config))
            time_budget = int(end_time - time.monotonic()) + 2  # wiggle room
            metadata_copy = deepcopy(metadata)
            metadata_copy.training_limit_sec = time_budget
            model = model_cls(metadata_copy, config, seed)
            if budget is None:
                model.fit(X_train, y_train, X_unlabeled, y_unlabeled, X_val, y_val)
            else:
                print('multi fidelity')
                budget_datapoints_labeled = int(len(y_train) * budget)
                budget_datapoints_unlabeled = int(len(y_unlabeled) * budget)
                model.fit(X_train[:budget_datapoints_labeled], y_train[:budget_datapoints_labeled],
                          X_unlabeled[:budget_datapoints_unlabeled], y_unlabeled[:budget_datapoints_unlabeled],
                          X_val, y_val)

            if metadata.evaluation_metric.value == 'bce':
                y_pred = model.predict_proba(X_test)[:, 1, :]
                model.load_model(get_model_save_path() / model.checkpoint_file)
                y_pred_cp = model.predict_proba(X_test)[:, 1, :]
            else:
                y_pred = model.predict(X_test)
                model.load_model(get_model_save_path() / model.checkpoint_file)
                y_pred_cp = model.predict(X_test)

            if metadata.output_type.value == "regression":
                shape = y_pred.shape
                y_pred = y_pred.reshape(len(y_pred), -1)
                y_pred = output_transformer.inverse_transform(y_pred)
                y_pred = y_pred.reshape(shape)

                y_pred_cp = y_pred_cp.reshape(len(y_pred_cp), -1)
                y_pred_cp = output_transformer.inverse_transform(y_pred_cp)
                y_pred_cp = y_pred_cp.reshape(shape)

            if suite in ['phase-2', 'phase-3'] and metadata.output_type.value == 'classification':
                if not is_multi_label(metadata):
                    y_pred = np.identity(math.prod(metadata.output_shape))[y_pred]
                    y_pred_cp = np.identity(math.prod(metadata.output_shape))[y_pred_cp]

            score = score_prediction(y_test, y_pred, metadata_copy)
            score_cp = score_prediction(y_test, y_pred_cp, metadata_copy)
            total_score += min(score, score_cp)

            del X_train
            del y_train
            del X_test
            del y_test
            del X_unlabeled
            del y_unlabeled
            del X_val
            del y_val
        total_score /= cv
        print('Finished training of config ', config)
        print('Config got a val score of ', total_score)

        return total_score

    return train_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs model on a single dataset.')
    parser.add_argument('--model', default='wrn', type=str, help="Specifies the model to run")
    parser.add_argument('--suite', default='phase-1', type=str, help="Benchmark suit of datasets")
    parser.add_argument('--dataset_id', default='listops', type=str,
                        help='The id of the dataset the model is run on')
    parser.add_argument('--dataset_dir', default=None, type=str, help="Directory where datasets are saved.")
    parser.add_argument('--time_budget', default=18000, type=int, help="The time hpo has to run")
    parser.add_argument('--n_trials', default=2, type=int, help="Number of maximum trials.")
    parser.add_argument('--n_workers', default=1, type=int, help="Number of parallel workers")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cv', default=1, type=int)
    parser.add_argument('--multi_fidelity', action="store_true")
    parser.add_argument('--run_name', default=None, type=str,
                        help="Optional parameter to further customize smac run name")

    args = parser.parse_args()

    model_cls = get_model(args.model)
    assert issubclass(model_cls, BaseModel)
    cs = model_cls.get_config_space()

    scenario_kwargs = {}
    if args.multi_fidelity:
        eta = 3
        scenario_kwargs['max_budget'] = 1.
        scenario_kwargs['min_budget'] = scenario_kwargs['max_budget'] / eta / eta

    output_directory = Path(__file__).parent / 'smac_output'
    output_directory.mkdir(exist_ok=True)

    opt_name = f'{args.suite}_{args.dataset_id}_{args.model}{"_" + args.run_name if args.run_name else ""}_{int(datetime.now().timestamp())}'
    scenario = Scenario(configspace=cs,
                        name=opt_name,
                        output_directory=output_directory,
                        walltime_limit=args.time_budget,
                        n_trials=args.n_trials,
                        n_workers=args.n_workers,
                        deterministic=True,
                        seed=args.seed,
                        **scenario_kwargs,
                        )

    train_function = get_train_function(model_cls, args.suite, args.dataset_id, args.dataset_dir, args.time_budget,
                                        args.cv)

    if args.multi_fidelity:
        intensifier = MultiFidelityFacade.get_intensifier(scenario, eta=eta)
        smac = MultiFidelityFacade(scenario=scenario, target_function=train_function, intensifier=intensifier)
    else:
        smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=train_function)  # add both
    incumbent = smac.optimize()
    print('Incumbent: ', incumbent)

    # dataset_result, trained_model = train_model(args.suite, args.model, args.dataset_id, args.dataset_dir,
    #                                             time_budget=args.time_budget,
    #                                             config=incumbent, seed=args.seed)
    # save_result(dataset_result, args.suite, args.dataset_id, args.model, trained_model)
