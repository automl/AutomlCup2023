from pathlib import Path

import numpy as np
from dataloader import AutoMLCupDataloader
from datasets import DatasetDict, load_dataset
from sklearn.preprocessing import OneHotEncoder


class SpliceDataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "splice"

    test_size = 0.1
    shuffle = False

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)

        cache_directory = directory / "cache"
        if cache_directory.exists():
            self.dataset = DatasetDict.load_from_disk(cache_directory)
        else:
            # character: meaning
            # D:  A or G or T
            # N:  A or G or C or T
            # S:  C or G
            # R:  A or G
            label_encoder = OneHotEncoder().fit([["N"], ["EI"], ["IE"]])
            character_classes = {
                "A": ["A", "D", "N", "R"],
                "C": ["C", "N", "S"],
                "G": ["G", "D", "N", "S", "R"],
                "T": ["T", "D", "N"],
                "D": ["D"],
                "N": ["N"],
                "S": ["S"],
                "R": ["R"],
            }
            character_encoder = {
                c: i for i, c in enumerate(["A", "C", "G", "T", "D", "N", "S", "R"])
            }

            def positions_to_vec(example):
                example["input"] = np.zeros((len(character_encoder) * 60,))

                index = 0
                for i in range(60):
                    characters = example[f"position_{i}"]
                    if characters == "nan":
                        continue
                    for character in characters:
                        all_character_classes = character_classes[character]
                        for character_class in all_character_classes:
                            example["input"][
                                index * len(character_encoder)
                                + character_encoder[character_class]
                            ] += 1
                        index += 1
                example["label"] = example["class"]

                return example

            self.dataset = (
                load_dataset(
                    "mstz/splice",
                    "splice",
                    data_dir=directory / SpliceDataloader.name(),
                )["train"]
                .map(
                    positions_to_vec,
                    remove_columns=list(f"position_{i}" for i in range(60)),
                )
                .remove_columns(["class"])
                .train_test_split(
                    test_size=SpliceDataloader.test_size,
                    shuffle=SpliceDataloader.shuffle,
                )
            )

            self.dataset.save_to_disk(cache_directory)

    def get_split(self, split):
        if split in ["train", "test"]:
            return self.dataset[split]
