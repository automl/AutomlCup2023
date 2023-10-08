import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from data.data_utils import get_dataset
from models.base_models.model_utils import get_model, is_multi_label, split_dataset, get_model_save_path, \
    get_eval_metric
from models.preprocessing import InputTransformer
from utils import deep_update

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train_model(suite, model_str, dataset_id, dataset_dir, time_budget, config=None, seed=0):
    metadata, output_transformer, X_train, y_train, X_test, y_test, X_unlabeled, y_unlabeled, X_val, y_val = \
        get_formatted_dataset(suite, dataset_id, dataset_dir)
    metadata.training_limit_sec = time_budget
    model_cls = get_model(model_str)
    model = model_cls(metadata, config, seed)
    model.fit(X_train, y_train, X_unlabeled, y_unlabeled, X_val, y_val)
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

    score = score_prediction(y_test, y_pred, metadata)
    score_cp = score_prediction(y_test, y_pred_cp, metadata)
    # roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovo") # needs probabilities
    renaming_dict = {'accuracy': 'misclassification rate'}
    evaluation_metric = renaming_dict.get(metadata.evaluation_metric.value, metadata.evaluation_metric.value)

    print(f'{model_str} reached an {evaluation_metric} of {score}.')
    print(f'{model_str} reached an {evaluation_metric} of {score_cp} for the checkpointed model.')

    dataset_result = {'model_name': model_str,
                      evaluation_metric: score,
                      evaluation_metric + ' check point': score_cp,
                      'date': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                      'seed': seed,
                      'config': model.cfg if hasattr(model, 'cfg') else None,
                      }

    if hasattr(model, 'run_statistics') and model.run_statistics is not None:
        dataset_result['run_statistics'] = model.run_statistics
    return dataset_result, model


def get_formatted_dataset(suite, dataset_id, dataset_dir, fold=0, cv=1):
    train_dataset, test_dataset, metadata = get_dataset(suite, dataset_id, dataset_dir)
    X_train, y_train = train_dataset['input'], train_dataset['label']
    X_test, y_test = test_dataset['input'], test_dataset['label']

    if cv > 1:
        labeled_datapoints = ~np.isnan(y_train).any(axis=tuple(range(1, y_train.ndim)))
        X_train, X_unlabeled = X_train[labeled_datapoints], X_train[~labeled_datapoints]
        y_train, y_unlabeled = y_train[labeled_datapoints], y_train[~labeled_datapoints]

        X_train = np.concatenate([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])
        kf = KFold(n_splits=cv, random_state=0, shuffle=True)
        train_index, test_index = [split for split in kf.split(X_train)][fold]
        X_train, X_test = X_train[train_index], X_train[test_index]
        y_train, y_test = y_train[train_index], y_train[test_index]

        X_train = np.concatenate([X_train, X_unlabeled])
        del X_unlabeled
        y_train = np.concatenate([y_train, y_unlabeled])
        del y_unlabeled

    output_transformer = StandardScaler()  # MinMaxScaler((-1, 1))
    if metadata.output_type.value == "regression":
        shape = y_train.shape
        y_train = y_train.reshape(len(y_train), -1)
        y_train = output_transformer.fit_transform(y_train)
        y_train = y_train.reshape(shape)

    if suite != 'phase-1':
        input_transformer = InputTransformer(metadata)
        metadata = input_transformer.fit(X_train)
        X_train = input_transformer.transform(X_train)
        X_test = input_transformer.transform(X_test)

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(y_train.shape + (1,))
        y_test = y_test.reshape(y_test.shape + (1,))

    if suite == 'phase-1':
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1, 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1, 1))
    elif suite in ['phase-2', 'phase-3'] and metadata.output_type.value == 'classification':
        if not is_multi_label(metadata):
            y_train = np.argmax(y_train, axis=1)

    labeled_datapoints = ~np.isnan(y_train).any(axis=tuple(range(1, y_train.ndim)))
    X_train, X_unlabeled = X_train[labeled_datapoints], X_train[~labeled_datapoints]
    y_train, y_unlabeled = y_train[labeled_datapoints], y_train[~labeled_datapoints]

    print("Loaded Dataset")

    X_train, X_val, y_train, y_val = split_dataset(X_train, y_train, metadata)
    return metadata, output_transformer, X_train, y_train, X_test, y_test, X_unlabeled, y_unlabeled, X_val, y_val


def score_prediction(y_test, y_pred, metadata):
    score_function = get_eval_metric(metadata)
    return score_function(y_test, y_pred)


def save_result(dataset_result, suite, dataset_id, model_str, trained_model):
    path = Path(__file__).parent / 'results' / 'simple_run' / f'{model_str}.json'
    path.parent.mkdir(parents=True, exist_ok=True)

    if trained_model is not None and hasattr(trained_model, 'save_model'):
        file_name = f'{model_str}_{int(datetime.now().timestamp())}.pkl'
        model_path = Path(__file__).parent / 'results' / 'trained_models' / file_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trained_model.save_model(str(model_path))
        if model_str == 'ensemble':
            # Update config since ensemble adds path to models to cfg when calling save_model
            dataset_result['config'] = trained_model.cfg
        else:
            dataset_result['model_file'] = file_name

    result = {suite: {dataset_id: [dataset_result]}}

    if path.exists():
        with path.open('r') as f:
            total_results = json.load(f)
            dataset_results = total_results.get(suite, {}).get(dataset_id, None)
            if dataset_results is None:
                total_results = deep_update(total_results, result)
            else:
                total_results[suite][dataset_id].append(dataset_result)
    else:
        total_results = result

    with path.open('w') as f:
        json.dump(total_results, f, indent=2)


def get_model_config(model_name: str, file_name: str):
    """Returns config loaded from configs/<model_name>/<file_name>"""
    if model_name is None or file_name is None:
        return None
    assert file_name.endswith('.json'), "Configurations have to be json files"
    file_path = Path(__file__).parent / 'configs' / model_name / file_name
    with file_path.open('r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs model on a single dataset.')
    parser.add_argument('--model', default="knn", type=str, help="Specifies the model to run")
    parser.add_argument('--suite', default='phase-1', type=str, help="Benchmark suit of datasets")
    parser.add_argument('--config_file', default=None, type=str, help="Name of config file in args.model directory.")
    parser.add_argument('--dataset_id', default='listops', type=str,
                        help='The id of the dataset the model is run on')
    parser.add_argument('--dataset_dir', default=None, type=str, help="Directory where datasets are saved")
    parser.add_argument('--time_budget', default=None, type=int, help="The time the model has to run")
    args = parser.parse_args()
    # add models directory to path to unify imports between ingestion and this run function
    sys.path.append('./models')
    args.model = args.model.lower()
    args.suite = args.suite.lower()
    model_config = get_model_config(args.model, args.config_file)
    dataset_result, trained_model = train_model(args.suite, args.model, args.dataset_id, args.dataset_dir,
                                                args.time_budget, model_config)
    save_result(dataset_result, args.suite, args.dataset_id, args.model, trained_model)
