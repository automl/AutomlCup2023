"""
    ListOps Dataloader.
    Adapted from: https://github.com/ctlllll/SGConv/blob/0c2b3ab/src/dataloaders/lra.py
"""

import logging
import pickle
from functools import partial
from pathlib import Path

import torch
import torchtext
from dataloader import AutoMLCupDataloader
from datasets import DatasetDict, load_dataset, concatenate_datasets


class SequenceDataset:
    registry = {}
    collate_args = []
    loader_registry = {
        None: torch.utils.data.DataLoader,  # default case
    }

    @property
    def init_defaults(self):
        """Default args"""
        raise NotImplementedError

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        super().__init__()
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        self.args = init_args

        # The train, val, test datasets must be set by `setup()`
        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.init()

    def init(self):
        """Hook called at end of __init__, override this instead of __init__"""

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test."""
        raise NotImplementedError

    def split_train_val(self, val_split):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def train_dataloader(self, **kwargs):
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None:
            return
        kwargs["shuffle"] = (
            "sampler" not in kwargs
        )  # shuffle cant be True if we have custom sampler
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None:
            return
        # Note that shuffle=False by default
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_

    @property
    def _name_(self):
        raise NotImplementedError

    @classmethod
    def _collate_callback(cls, x, *_args, **_kwargs):
        """
        Modify the behavior of the default _collate method.
        """
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *_args, **_kwargs):
        """
        Modify the return value of the collate_fn.
        Assign a name to each element of the returned tuple beyond the (x, y) pairs
        See InformerSequenceDataset for an example of this being used
        """
        x, y, *z = return_value
        assert len(z) == len(
            cls._collate_arg_names
        ), "Specify a name for each auxiliary data item returned by dataset"
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        # From https://github.com/pyforch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                # pylint: disable-next=protected-access
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)

            # Insert custom functionality into the collate_fn
            x = cls._collate_callback(x, *args, **kwargs)

            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Default collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    # List of loader arguments to pass into collate_fn
    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {
            k: loader_args[k] for k in loader_args if k in self.collate_args
        }
        loader_args = {
            k: loader_args[k] for k in loader_args if k not in self.collate_args
        }
        loader_cls = self.loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )


class ListOpsDataset(SequenceDataset):
    d_output = 10
    l_output = 0

    def __init__(self, *args, directory: Path = None, **kwargs):
        if not isinstance(directory, Path):
            raise ValueError(f"Must provide directory for {__class__.__name__}.")
        self.directory = directory

        super().__init__(self._name_, *args, **kwargs)
        self.tokenizer = None
        self.vocab = None
        self.vocab_size = None
        self._collate_fn = None

    @property
    def _name_(self):
        return "listops"

    @property
    def init_defaults(self):
        return {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return (
            f"l_max-{self.args['l_max']}"
            f"-append_bos-{self.args['append_bos']}"
            f"-append_eos-{self.args['append_eos']}"
        )

    def init(self):
        if self.data_dir is None:
            self.data_dir = self.directory / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = torch.nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        tokenizer = listops_tokenizer

        # Account for <bos> and <eos> tokens
        l_max = (
            self.args["l_max"]
            - int(self.args["append_bos"])
            - int(self.args["append_eos"])
        )
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.args["n_workers"], 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.args["append_bos"] else [])
                + (["<eos>"] if self.args["append_eos"] else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.args["append_bos"] else [])
                + example["tokens"]
                + (["<eos>"] if self.args["append_eos"] else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.args["n_workers"], 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info("Saving to cache at %s", str(cache_dir))
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info("Load from cache at %s", str(cache_dir))
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class ListOpsDataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "listops"

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)
        listops_dataset = ListOpsDataset(directory=directory)
        dataset: DatasetDict
        (dataset, _tokenizer, _vocab) = listops_dataset.process_dataset()

        self.dataset = dataset.rename_columns(
            {
                "input_ids": "input",
                "Target": "label",
            }
        )

    def get_split(self, split):
        if split == "test":
            return self.dataset["test"]
        if split == "train":
            return concatenate_datasets([self.dataset["train"], self.dataset["val"]])


# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()
