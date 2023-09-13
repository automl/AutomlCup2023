from pathlib import Path
from abc import ABC, abstractmethod
from datasets import Dataset


class AutoMLCupDataloader(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    def __init__(self, directory: Path, **_kwargs) -> None:
        self.directory = directory

    @abstractmethod
    def get_split(self, split: str) -> Dataset:
        raise NotImplementedError
