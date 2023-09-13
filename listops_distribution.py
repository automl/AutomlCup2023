import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.data_utils import get_dataset
from models.model_utils import get_model
from run_config import get_model_config


def plot_length_distribution(correct_X, incorrect_X, save_dir):
    fig, ax = plt.subplots()
    len_correct_X = (correct_X != 0).sum(axis=1)
    len_incorrect_X = (incorrect_X != 0).sum(axis=1)
    ax.hist(len_correct_X, bins=15, density=True, label='Correctly classified', alpha=0.5)
    ax.hist(len_incorrect_X, bins=15, density=True, label='Incorrectly classified', alpha=0.5)
    ax.legend(loc='upper right')
    ax.set_title('Length Distribution of Test Data')
    ax.set_xlabel('length')
    ax.set_xlabel('frequency')
    fig.savefig(save_dir / 'length_distribution.png')


def plot_label_frequency(y_test, y_pred, save_dir):
    fig, ax = plt.subplots()
    x_ticks = np.arange(0, 10, 1)
    bins = list(np.linspace(-0.5, 9.5, 11))
    ax.hist(y_test, bins=bins, density=True, label='true labels', alpha=0.5)
    ax.hist(y_pred, bins=bins, density=True, label='predicted labels', alpha=0.5)
    ax.set_xticks(x_ticks)
    ax.set_title('Label Frequency')
    ax.legend(loc='upper center')
    ax.set_xlabel('target class')
    ax.set_xlabel('frequency')
    fig.savefig(save_dir / 'label_frequency.png')


def plot_prediction_metrics(correct_y, incorrect_y, y_test, save_dir):
    labels = sorted(list(set(y_test)))
    label_statistics = {
        'Recall/True Positive Rate': [],
        'Precision': [],
        'F1 Score': [],
    }
    for label in labels:
        true_labels = np.sum(y_test == label)
        true_positive = np.sum(correct_y == label)
        false_positive = np.sum(incorrect_y == label)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / true_labels
        f1_score = 2 * (precision * recall) / (precision + recall)
        label_statistics['Precision'].append(round(precision, 2))
        label_statistics['Recall/True Positive Rate'].append(round(recall, 2))
        label_statistics['F1 Score'].append(round(f1_score, 2))

    fig, ax = plt.subplots(layout='constrained', figsize=(13, 5))
    create_bar_chart(labels, label_statistics, ax)
    ax.set_xlabel('target classes')
    ax.set_title('Prediction Metrics for Target Classes')
    ax.legend(loc='upper center')
    fig.savefig(save_dir / 'prediction_metrics_target.png')


def plot_input_metrics(correct_x, incorrect_x, save_dir):
    idx = 0
    correct_x, incorrect_x = correct_x[:, idx], incorrect_x[:, idx]
    total_entries = len(correct_x) + len(incorrect_x)
    labels = sorted(list(set(np.append(correct_x, incorrect_x))))

    label_statistics = {
        'Correctly Classified': [],
        'Occurence': [],
    }
    for label in labels:
        correctly_classified = np.sum(correct_x == label)
        incorrectly_classified = np.sum(incorrect_x == label)
        occurence = (correctly_classified + incorrectly_classified) / total_entries
        ratio_correctly_classified = correctly_classified / (correctly_classified + incorrectly_classified)
        label_statistics['Correctly Classified'].append(round(ratio_correctly_classified, 2))
        label_statistics['Occurence'].append(round(occurence, 2))
    fig, ax = plt.subplots(layout='constrained')
    label_decoder = {'14': 'MIN', '15': 'SM', '16': 'MAX', '17': 'MED'}

    labels = [label_decoder.get(str(int(label)), label) for label in labels]
    create_bar_chart(labels, label_statistics, ax)
    ax.set_xlabel('operation')
    ax.set_title('Prediction Metrics - First Entry of Input')
    fig.savefig(save_dir / 'prediction_metrics_input.png')


def create_bar_chart(labels, values, ax):
    x = np.arange(len(labels))  # the label locations
    width = 0.275  # the width of the bars
    multiplier = 0

    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)  # ,label_type='center', rotation=90
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x + width * (len(values) - 1.) / 2, labels)
    ax.legend()
    ax.set_ylim(0., 1.)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE', linewidth=1.5)
    ax.xaxis.grid(False)
    ax.set_ylabel('frequency')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs model on a single dataset.')
    parser.add_argument('--model', default="wrn", type=str, help="Specifies the model to run")
    parser.add_argument('--suite', default='phase-1', type=str, help="Benchmark suit of datasets")
    parser.add_argument('--config_file', default=None, type=str, help="Name of config file in args.model directory.")
    parser.add_argument('--dataset_id', default='listops', type=str,
                        help='The id of the dataset the model is run on')
    parser.add_argument('--dataset_dir', default=None, type=str, help="Directory where datasets are saved")
    parser.add_argument('--time_budget', default=None, type=int, help="The time the model has to run")
    args = parser.parse_args()

    args.model = args.model.lower()
    args.suite = args.suite.lower()
    model_config = get_model_config(args.model, args.config_file)

    train_dataset, test_dataset, metadata = get_dataset(args.suite, args.dataset_id, args.dataset_dir)
    model_cls = get_model(args.model, args.suite)
    X, y = train_dataset['input'], train_dataset['label']
    model = model_cls(X.shape[1:], y.shape[1:], np.max(y) + 1, args.time_budget, model_config, 0)
    model.load_model('./results/trained_models/wrn_1690501067.pkl')
    X_test = test_dataset['input']
    y_pred = model.predict(X_test)
    y_test = test_dataset['label']
    mask = y_test == y_pred
    correct_X = X_test[mask]
    incorrect_X = X_test[~mask]
    correct_y = y_pred[mask]
    incorrect_y = y_pred[~mask]

    save_dir = path = Path(
        __file__).parent / 'images' / 'listops_distribution' / f"{args.model}_{args.config_file.replace('.json', '')}"
    path.mkdir(parents=True, exist_ok=True)

    plot_length_distribution(correct_X, incorrect_X, save_dir)
    plot_label_frequency(y_test, y_pred, save_dir)
    plot_prediction_metrics(correct_y, incorrect_y, y_test, save_dir)
    plot_input_metrics(correct_X, incorrect_X, save_dir)
