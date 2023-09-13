# This is a sample code submission.
# It is a simple machine learning classifier.
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim))

    def forward(self, x):
        return self.classifier(x)


def get_dataloader(X, y, batch_size: int, shuffle: bool):
    X = torch.from_numpy(X.astype("float32"))  # .astype("float32")
    y = torch.from_numpy(y)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    return dataloader


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

        self.classifier = MLP(X.shape[1], np.max(y)+1)
        self.classifier.to(self.device)
        optimizer = AdamW(self.classifier.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()  # BCEWithLogitsLoss()
        self.classifier.train()
        epochs = 100
        for _ in range(epochs):
            for data, target in get_dataloader(X, y, 100, True):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                optimizer.zero_grad()
                output = self.classifier(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

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

        y = self.classifier(torch.from_numpy(X.astype("float32")).to(self.device))  # should be a loop
        _, y = torch.max(y, 1)
        return y.detach().cpu().numpy()
