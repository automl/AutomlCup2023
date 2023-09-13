# This is a sample code submission.
# It is a simple machine learning classifier.
import numpy as np
import torch.nn as nn
from datasets import Dataset
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, WarmRestartLR
from torch.optim import AdamW


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim))

    def forward(self, x):
        return self.classifier(x)


class Model:
    def __init__(self):
        """<ADD DOCUMENTATION HERE>"""
        self.classifier = None
        self.device = 'cuda'

    def train(self, train_dataset: Dataset):
        """Train the model.

        Args:
            train_dataset: Training data, type datasets.Dataset.
        """
        print(f"FROM MODEL.PY: TRAIN {train_dataset}")

        try:
            X = np.array(train_dataset["input"])
        except ValueError:
            max_len = max(map(len, train_dataset["input"]))
            X = np.zeros((len(train_dataset), max_len), dtype=np.float32)
            for idx, row in enumerate(train_dataset["input"]):
                X[idx, : len(row)] = row

        y = np.array(train_dataset["label"])
        # use callbacks to inject early stopping, scoring or schedulers
        callbacks = []
        callbacks.append(EarlyStopping(monitor="valid_loss"))  # , patience=10
        callbacks.append(LRScheduler(WarmRestartLR, min_lr=2e-5, max_lr=1e-2))
        # callbacks.append(EpochScoring(scoring="accuracy", name="train_accuracy", on_train=True))

        self.classifier = NeuralNetClassifier(
            MLP,
            optimizer=AdamW,
            batch_size=100,
            module__input_dim=X.shape[1],
            module__output_dim=np.max(y) + 1,
            max_epochs=30,
            criterion=nn.CrossEntropyLoss(),
            lr=1e-3,
            device='cuda',
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks=callbacks,
        )
        self.classifier.fit(X.astype("float32"), y)

    def predict(self, dataset: Dataset):
        """Predict labels.

        Args:
            test_dataset: Testing data, type datasets.Dataset.
        """
        print(f"FROM MODEL.PY: TEST {dataset}")
        try:
            X = np.array(dataset["input"])
        except ValueError:
            max_len = max(map(len, dataset["input"]))
            X = np.zeros((len(dataset), max_len), dtype=np.float32)
            for idx, row in enumerate(dataset["input"]):
                X[idx, : len(row)] = row

        y = self.classifier.predict(X.astype("float32"))
        return y
