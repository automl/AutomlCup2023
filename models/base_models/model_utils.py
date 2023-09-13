# TODO: maybe use soft f1 instead
from pathlib import Path
from typing import Union

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nn
from torch.nn.functional import binary_cross_entropy, l1_loss, mse_loss


# https://discuss.pytorch.org/t/rmse-loss-function/16540/4
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def get_model_save_path():
    trained_models_path = Path(__file__).parent / 'trained_models'
    trained_models_path.mkdir(exist_ok=True, parents=True)
    return trained_models_path


def get_loss_function(metadata):
    eval_metric_name = metadata.evaluation_metric.value
    if eval_metric_name == "accuracy":
        return nn.CrossEntropyLoss()
    if eval_metric_name == "mae":
        return nn.L1Loss()
    if eval_metric_name == "bce":
        return nn.BCEWithLogitsLoss()
    if eval_metric_name == "rmse":
        return RMSELoss()  # untested
    if eval_metric_name == "f1":
        # maybe use https://stackoverflow.com/a/76842670 instead
        return nn.BCEWithLogitsLoss()
    if eval_metric_name == "mse":
        return nn.MSELoss()


def get_eval_metric(metadata):
    eval_metric_name = metadata.evaluation_metric.value
    if eval_metric_name == "accuracy":
        return lambda x, y: 1 - accuracy_score(x, y)
    if eval_metric_name == "mae":
        return lambda x, y: l1_loss(Tensor(y), Tensor(x), reduction="mean").item()
    if eval_metric_name == "bce":
        return lambda x, y: binary_cross_entropy(Tensor(y), Tensor(x), reduction="mean").item()
    if eval_metric_name == "rmse":
        return lambda x, y: torch.sqrt(mse_loss(Tensor(y), Tensor(x), reduction="mean")).item()
    if eval_metric_name == "f1":
        return lambda x, y: 1 - f1_score(x, y)
    if eval_metric_name == "mse":
        return lambda x, y: mse_loss(Tensor(y), Tensor(x), reduction="mean").item()


def get_model(model_str: str):
    model_str = model_str.lower()
    if model_str == "wrn":
        from .wrn_model import WRN_Model
        model_cls = WRN_Model
    elif model_str == "unet":
        from .unet import UNET_Model
        model_cls = UNET_Model
    elif model_str == "ensemble":
        from .ensemble_model import Ensemble_Model
        model_cls = Ensemble_Model
    elif model_str == "multires":
        from .multires_conv_model import MultiResConvModel
        model_cls = MultiResConvModel
    elif model_str == "seq_transformer":
        from .sequence_transformer_model import SequenceTransformer_Model
        model_cls = SequenceTransformer_Model
    elif model_str == "efn":
        from .efn_model import EFN_Model
        model_cls = EFN_Model
    elif model_str == "vit":
        from .vit_model import VIT_Model
        model_cls = VIT_Model
    elif model_str == "mlp":
        from .mlp_model import MLP_Model
        model_cls = MLP_Model
    else:
        raise ValueError(f"The model name '{model_str}' is unknown. Use a known model name")
    return model_cls


def is_multi_label(metadata):
    return metadata.evaluation_metric.value in ['f1', 'bce']


def split_dataset(X, y, metadata):
    stratify_kwargs = {}
    if metadata.output_type.value == 'classification' and metadata.evaluation_metric.value != 'bce':
        stratify_kwargs['stratify'] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, **stratify_kwargs)
    return X_train, X_test, y_train, y_test


def instantiate_model(model_name: str, metadata, config: Union[list, dict] = None, seed=42):
    # Todo: might be deprecated
    assert model_name != 'ensemble' or config is None or type(config) is list, \
        'Configs for ensemble model must be list or None.'
    assert model_name == 'ensemble' or config is None or type(config) is dict, \
        'Configs for non ensemble models must be dict or None'

    model_cls = get_model(model_name)
    return model_cls(metadata, config, seed)
