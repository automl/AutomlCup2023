"""AutoML datasets."""

import json
from pathlib import Path
from typing import List, Optional, Type, TypeVar

from splice import SpliceDataloader

from common import get_logger
from dataloader import AutoMLCupDataloader
from datasets import Dataset
from listops import ListOpsDataloader

VERBOSITY_LEVEL = "WARNING"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class AutoMLCupDataset:
    """AutoMLCupDataset"""

    D = TypeVar("D", bound=AutoMLCupDataloader)
    dataloaders: List[Type[D]] = [ListOpsDataloader, SpliceDataloader]

    def __init__(self, directory: Path):
        """init"""
        dataset: Optional[AutoMLCupDataloader] = None

        with open(directory / "info.json", encoding="utf-8") as json_file:
            dataset_info = json.load(json_file)
            dataset_name = dataset_info["name"]

        for dataloader in AutoMLCupDataset.dataloaders:
            if dataset_name == dataloader.name():
                dataset = dataloader(directory)
                break

        if dataset is None:
            raise ValueError(f"Dataset from {directory} not found.")
        self.dataset = dataset

    def name(self) -> str:
        return self.dataset.name()

    def get_split(self, split: str) -> Dataset:
        return self.dataset.get_split(split)
