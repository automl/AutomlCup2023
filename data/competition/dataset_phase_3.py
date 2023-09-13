"""AutoML datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

from numpy.typing import ArrayLike

from .common import get_logger
from .dataloader import AutoMLCupDataloader
from .metadata import AutoMLCupMetadata, InputShape, OutputType, EvaluationMetric

from .pde import PDEDiffusionloader
from .camelyon17 import Camelyon17Dataloader
from .globalwheat import GlobalWheatDataloader

VERBOSITY_LEVEL = "WARNING"
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class AutoMLCupDatasetPhase3:
    """AutoMLCupDataset"""

    D = TypeVar("D", bound=AutoMLCupDataloader)
    dataloaders: List[Type[D]] = [
        PDEDiffusionloader,
        Camelyon17Dataloader,
        GlobalWheatDataloader,
    ]

    def __init__(self, directory: Path, datasets_root: Path):
        """init"""
        dataset: Optional[AutoMLCupDataloader] = None

        with open(directory / "info.json", encoding="utf-8") as json_file:
            dataset_info = json.load(json_file)
            dataset_name = dataset_info["name"]
            self._metadata = AutoMLCupMetadata(
                dataset_info["input_dimension"],
                InputShape(
                    dataset_info["num_examples"],
                    dataset_info["max_sequence_len"],
                    dataset_info["channels"],
                    dataset_info["width"],
                    dataset_info["height"],
                ),
                dataset_info["output_shape"],
                OutputType(dataset_info["output_type"]),
                EvaluationMetric(dataset_info["evaluation_metric"]),
                dataset_info["training_limit_sec"],
            )

        for dataloader in AutoMLCupDatasetPhase3.dataloaders:
            if dataset_name == dataloader.name():
                dataset = dataloader(datasets_root / dataset_name)
                break

        if dataset is None:
            raise ValueError(f"Dataset from {directory} not found.")
        self.dataset = dataset

    def name(self) -> str:
        return self.dataset.name()

    def get_split(self, split: str) -> Dict[str, ArrayLike]:
        return self.dataset.get_split(split)

    def metadata(self) -> AutoMLCupMetadata:
        return self._metadata