import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs model on a single dataset.')
    parser.add_argument('--dataset', default='protein', type=str, help="Specifies the model to run")
    args = parser.parse_args()

    path = Path(__file__).parent / 'datasets' / 'competition' / args.dataset
    X = np.load(path / 'x_original.npz')['x_train']
    y = np.load(path / 'y_original.npz')['y_train']

    with open(path / 'info.json', 'r') as f:
        info = json.load(f)
    kwargs = {}
    if info['output_type'] == 'classification':
        kwargs['stratify'] = y
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, **kwargs)
    np.savez(path / 'x_train.npz', x_train=x_train)
    np.savez(path / 'y_train.npz', y_train=y_train)
    np.savez(path / 'x_test.npz', x_test=x_test)
    np.savez(path / 'y_test.npz', y_test=y_test)
