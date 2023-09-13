import argparse
import os
from pathlib import Path

import numpy as np

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from data.competition.dataset_phase_1 import AutoMLCupDatasetPhase1
from data.competition.dataset_phase_2 import AutoMLCupDatasetPhase2
from data.competition.dataset_phase_3 import AutoMLCupDatasetPhase3
from models.p1metadata import AutoMLCupMetadata, InputShape, EvaluationMetric, OutputType


def load_openml(did, root):
    try:
        import openml
    except:
        os.system("pip install openml")
        import openml

    fname = root + f'/{did}_dev_test_split.npy'
    if not os.path.isfile(fname):
        prepare_data(root, did)

    data = np.load(fname, allow_pickle=True)
    data = data.item()
    X_train, X_test = data['X_dev'], data['X_test']
    y_train, y_test = data['y_dev'], data['y_test']

    train_dataset = dummy_dataset(X_train, y_train)
    test_dataset = dummy_dataset(X_test, y_test)

    return train_dataset, test_dataset


def prepare_data(root, did):
    X, y = load_openml_dataset(did)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

    count = 0
    for dev_index, test_index in sss.split(X, y):
        assert count == 0
        X_dev, X_test = X[dev_index], X[test_index]
        y_dev, y_test = y[dev_index], y[test_index]
        count += 1

    data = {'y_dev': y_dev, 'y_test': y_test, 'X_dev': X_dev, 'X_test': X_test}

    np.save(root + f'/{did}_dev_test_split.npy', data)


def load_openml_dataset(did=61):
    import openml
    ds = openml.datasets.get_dataset(did)
    # values
    X, y, categorical_indicator, attribute_names = ds.get_data(target=ds.default_target_attribute)
    # preprocess
    Xy = pd.concat([X, y], axis=1, ignore_index=True)  # X & y concatenated together

    Xy.replace('?', np.NaN, inplace=True)  # replace ? with NaNs
    Xy = Xy[Xy.iloc[:, -1].notna()]  # remove all the rows whose labels are NaN
    y_after_NaN_removal = Xy.iloc[:, -1]
    Xy.dropna(axis=1, inplace=True)  # drop all the columns with missing entries
    assert ((Xy.iloc[:, -1] == y_after_NaN_removal).all())
    X_raw, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]
    categorical_indicator = [categorical_indicator[i] for i in Xy.columns[:-1]]

    # fine the categorical
    categorial_indices = np.where(np.array(categorical_indicator))[0]
    noncat_indices = np.where(np.invert(np.array(categorical_indicator)))[0]
    scaler = StandardScaler()

    if len(categorial_indices) > 0:
        enc = OneHotEncoder()  # handle_unknown='ignore'
        X_norm = enc.fit_transform(X_raw.iloc[:, categorial_indices]).toarray()
        if len(noncat_indices) > 0:
            X_noncat = scaler.fit_transform(X_raw.iloc[:, noncat_indices])
            X_norm = np.concatenate((X_norm, X_noncat), axis=1)
    else:
        X_norm = scaler.fit_transform(X_raw)  # for only numerical

    try:
        y = y.cat.codes.values
    except:
        y = np.array(y.values).astype(int)

    print("cat feat:", len(categorial_indices), "numerical feat:", len(noncat_indices), "X shape", X.shape,
          X_norm.shape, "y shape", y.shape)

    return X_norm, y


def listops_histogram(dataset_dir, dataset_type):
    assert 'listops' in str(dataset_dir)
    import matplotlib.pyplot as plt
    cup_dataset = AutoMLCupDatasetPhase1(dataset_dir)
    dataset = cup_dataset.get_split(dataset_type)
    lengths = [len(row) for row in dataset['input']]
    plt.hist(np.array(lengths), bins='auto')
    plt.show()


def save_listops_small(dataset_dir, dataset_type, length):
    assert 'listops' in str(dataset_dir)
    cup_dataset = AutoMLCupDatasetPhase1(dataset_dir)
    dataset = cup_dataset.get_split(dataset_type)
    X = [(idx, row) for idx, row in enumerate(dataset['input']) if len(row) <= length]
    idx, X = list(zip(*X))
    y = np.array(dataset['label'])[np.array(idx)]
    dataset = dummy_dataset(X, y)
    X, y = convert_cup_dataset(dataset)
    save_dir = dataset_dir / 'processed_datasets'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f'X_{dataset_type}_{length}', X)
    np.save(save_dir / f'y_{dataset_type}_{length}', y)
    print(f'There are {len(y)} datapoins with length less or equal than {length}')


def load_automl_cup_phase_2(dataset_dir):
    cup_dataset = AutoMLCupDatasetPhase2(dataset_dir)
    train = cup_dataset.get_split('train')
    test = cup_dataset.get_split('test')

    X_train, y_train = convert_cup_dataset(train)
    X_test, y_test = convert_cup_dataset(test)

    train_dataset = dummy_dataset(X_train, y_train)
    test_dataset = dummy_dataset(X_test, y_test)

    return train_dataset, test_dataset, cup_dataset.metadata()

def load_automl_cup_phase_3(dataset_dir: Path):
    cup_dataset = AutoMLCupDatasetPhase3(dataset_dir, dataset_dir.parent)
    train = cup_dataset.get_split('train')
    test = cup_dataset.get_split('test')

    X_train, y_train = convert_cup_dataset(train)
    X_test, y_test = convert_cup_dataset(test)

    train_dataset = dummy_dataset(X_train, y_train)
    test_dataset = dummy_dataset(X_test, y_test)

    return train_dataset, test_dataset, cup_dataset.metadata()


def load_automl_cup_phase_1(dataset_dir):
    try:
        X_train = np.load(dataset_dir / 'X_train.npy')
        X_test = np.load(dataset_dir / 'X_test.npy')
        y_train = np.load(dataset_dir / 'y_train.npy')
        y_test = np.load(dataset_dir / 'y_test.npy')

        metadata = AutoMLCupMetadata(1, int(np.max(X_train) + 1),
                                     InputShape(X_train.shape[0], X_train.shape[1], 1, 1, 1), [np.max(y_train) + 1],
                                     OutputType("classification"), EvaluationMetric("accuracy"), None)

        train_dataset = dummy_dataset(X_train, y_train)
        test_dataset = dummy_dataset(X_test, y_test)
        return train_dataset, test_dataset, metadata
    except FileNotFoundError:
        pass

    cup_dataset = AutoMLCupDatasetPhase1(dataset_dir)
    train = cup_dataset.get_split('train')
    test = cup_dataset.get_split('test')
    X_train, y_train = convert_cup_dataset(train)
    X_test, y_test = convert_cup_dataset(test)

    np.save(dataset_dir / 'X_train.npy', X_train)
    np.save(dataset_dir / 'X_test.npy', X_test)
    np.save(dataset_dir / 'y_train.npy', y_train)
    np.save(dataset_dir / 'y_test.npy', y_test)

    metadata = AutoMLCupMetadata(1, InputShape(X_train.shape[0], X_train.shape[1], 1, 1, 1), [np.max(y_train) + 1],
                                 OutputType("classification"), EvaluationMetric("accuracy"), None)

    train_dataset = dummy_dataset(X_train, y_train)
    test_dataset = dummy_dataset(X_test, y_test)

    return train_dataset, test_dataset, metadata


def convert_cup_dataset(cup_dataset):
    try:
        X = np.array(cup_dataset["input"])
    except ValueError:
        max_len = max(map(len, cup_dataset["input"]))
        X = np.zeros((len(cup_dataset['label']), max_len), dtype=np.float32)
        for idx, row in enumerate(cup_dataset["input"]):
            X[idx, : len(row)] = row

    y = np.array(cup_dataset["label"])
    # X, y = np.array(cup_dataset["input"]), np.array(cup_dataset["label"])
    return X, y


def dummy_dataset(x, y):
    dataset = {'input': x, 'label': y}
    return dataset


if __name__ == "__main__":
    """
    Visualize dimensions of openml-cc18 and safe it as a csv.
    """
    parser = argparse.ArgumentParser(description='Runs model on a single dataset.')
    parser.add_argument('--method', default='cc18_summary', type=str,
                        help="Specifies what methods to run. Either 'cc18_summary', 'listops_hist' or 'listops_subset'")
    parser.add_argument('--dataset_type', default='train', type=str,
                        help="If method 'listops_hist' or 'listops_subset'")
    parser.add_argument('--length', default=600, type=int, help="If method 'listops_subset'")
    args = parser.parse_args()
    assert args.method in ['cc18_summary', 'listops_hist', 'listops_subset']

    if args.method == 'listops_hist':
        dataset_dir = Path(__file__).parent / 'datasets' / 'competition' / 'listops'
        listops_histogram(dataset_dir, args.dataset_type)
    elif args.method == 'listops_subset':
        dataset_dir = Path(__file__).parent / 'datasets' / 'competition' / 'listops'
        save_listops_small(dataset_dir, args.dataset_type, args.length)
    elif args.method == 'cc18_summary':
        import openml
        suite = openml.study.get_suite('OpenML-CC18')

        input_shape = []
        output_dimension = []
        task_ids = []
        num_datapoints = []
        task_types = []
        dataset_ids = []
        root = Path(__file__).parent / 'datasets' / 'openml'
        root.mkdir(parents=True, exist_ok=True)

        for task_id in suite.tasks:
            if task_id == 167124:
                continue
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset_id = task.dataset_id

            train_dataset, test_dataset = load_openml(dataset_id, str(root))
            task_ids.append(task_id)
            num_datapoints.append(train_dataset['input'].shape[0])
            output_dimension.append(train_dataset['label'].shape)
            input_shape.append(train_dataset['input'].shape[1:])
            task_types.append(task.task_type)
            dataset_ids.append(dataset_id)

        summary_df = pd.DataFrame(data={"dataset id": dataset_ids,
                                        "task id": task_ids,
                                        "number datapoints": num_datapoints,
                                        "input shape": input_shape,
                                        "output shape": output_dimension,
                                        "task type": task_types})
        summary_df.to_csv(str(root.parent / "openml_summary.csv"))
