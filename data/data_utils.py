import argparse
from pathlib import Path


def get_dataset(suite, dataset_id, root=None):
    suite = suite.lower()
    if root is None:
        root = '/work/dlclarge2/hogj-automlcup/AutoMLCup/data/datasets/'
    metadata = None
    if suite == 'openml-cc18':
        from data.data_loaders import load_openml
        train_dataset, test_dataset = load_openml(dataset_id, root + 'openml')
    elif suite == 'phase-1':
        from data.data_loaders import load_automl_cup_phase_1
        dataset_dir = Path(root) / 'competition' / dataset_id
        train_dataset, test_dataset, metadata = load_automl_cup_phase_1(dataset_dir)
    elif suite == 'phase-2':
        from data.data_loaders import load_automl_cup_phase_2
        dataset_dir = Path(root) / 'competition' / dataset_id
        train_dataset, test_dataset, metadata = load_automl_cup_phase_2(dataset_dir)
    elif suite == 'phase-3':
        from data.data_loaders import load_automl_cup_phase_3
        dataset_dir = Path(root) / 'competition' / dataset_id
        train_dataset, test_dataset, metadata = load_automl_cup_phase_3(dataset_dir)
    else:
        raise ValueError(f'{suite} is not a benchmark suite that is implemented.')
    return train_dataset, test_dataset, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs model on a single dataset.')
    parser.add_argument('--suite', default='phase-2', type=str)
    parser.add_argument('--dataset_id', default='splice', type=str)
    args = parser.parse_args()

    path = Path(__file__).parent / 'datasets'
    train_dataset, test_dataset, metadata = get_dataset('phase-2', 'splice', path)
    print('Done')