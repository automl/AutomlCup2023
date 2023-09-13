"""
Reference : https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
"""
import math

import numpy as np
import torch.nn as nn
import torchvision.models as models
from ConfigSpace import EqualsCondition, UniformFloatHyperparameter, CategoricalHyperparameter, Constant, \
    UniformIntegerHyperparameter
from skorch.callbacks import LRScheduler
from torch.optim import Adam, AdamW
from torchvision.models import EfficientNet_B0_Weights

from .skorch_base import SkorchBaseModel
from .unet import UNET_Model


class EFN(nn.Module):

    def __init__(self, metadata, num_classes, use_pretrained=True, dropout=0.3, num_layers=2, activation_ff='gelu',
                 multi_dropout=False):
        super(EFN, self).__init__()
        self.metadata = metadata
        timeseries = self.metadata.input_shape.max_sequence_len
        channel = self.metadata.input_shape.channels
        self.net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if use_pretrained else None)
        num_ftrs = self.net.classifier[1].in_features
        self.net.classifier[0].p = dropout
        last_layers = []
        layer_dims = np.rint(np.logspace(np.log10(num_classes), np.log10(num_ftrs), num_layers + 1))[::-1].astype(int)
        for i in range(num_layers):
            last_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < num_layers - 1:  # not last layer of last layers
                if activation_ff == 'gelu':
                    last_layers.append(nn.GELU())
                elif activation_ff == 'silu':
                    last_layers.append(nn.SiLU())
                if multi_dropout:
                    last_layers.append(nn.Dropout(dropout))

        self.net.classifier[1] = nn.Sequential(*last_layers)
        if not (timeseries * channel == 1 or timeseries * channel == 3):
            self.net.features[0][0] = nn.Conv2d(timeseries * channel, 32, kernel_size=(3, 3), stride=(2, 2),
                                                padding=(1, 1), bias=False)

    def forward(self, x):
        # TODO fix for timeseries > 1 and channel=1 and make sure its correct in all cases 
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        x = x.reshape((x.shape[0], -1, x.shape[3], x.shape[4]))
        x = self.net.forward(x)
        return x

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        for parameter in model.parameters():
            parameter.requires_grad = False


class EFN_Model(SkorchBaseModel):
    def __init__(self, metadata, input_cfg=None, seed=None):
        super().__init__(metadata, input_cfg, seed)
        if self.cfg['optimizer'] == 'Adam':
            self.skorch_kwargs['optimizer'] = Adam
        elif self.cfg['optimizer'] == 'AdamW':
            self.skorch_kwargs['optimizer'] = AdamW
            self.skorch_kwargs['optimizer__weight_decay'] = self.cfg['weight_decay']

        if self.cfg['lr_scheduler']:
            self.callbacks.append(
                self.callbacks.append(LRScheduler(policy='CosineAnnealingLR', T_max=self.cfg['max_epochs'])))

        self.model = self.model_class(
            EFN,
            module__metadata=metadata,
            module__dropout=self.cfg['dropout'],
            module__num_classes=math.prod(self.output_shape),
            module__use_pretrained=self.cfg['pretrained'],
            module__multi_dropout=self.cfg['multi_dropout'],
            module__activation_ff=self.cfg['activation_ff'],
            module__num_layers=self.cfg['num_layers'],
            max_epochs=self.cfg['max_epochs'],
            lr=self.cfg['lr'],
            callbacks=self.callbacks,
            **self.skorch_kwargs
        )

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="efn")
        lr = UniformFloatHyperparameter(
            name="lr", lower=1e-5, upper=1e-2, default_value=1e-3, log=True
        )
        optimizer = CategoricalHyperparameter(
            name="optimizer", choices=['Adam', 'AdamW'], default_value='AdamW'
        )
        lr_scheduler = CategoricalHyperparameter(
            name="lr_scheduler", choices=[True, False], default_value=True
        )
        weight_decay = UniformFloatHyperparameter(
            name="weight_decay", lower=1e-7, upper=1e-1, default_value=0.03, log=True
        )
        dropout = UniformFloatHyperparameter(
            name="dropout", lower=0., upper=0.5, default_value=0.2, log=False
        )
        max_epochs = UniformIntegerHyperparameter(
            name="max_epochs", lower=1, upper=300, default_value=300, log=False
        )
        pretrained = CategoricalHyperparameter(
            name="pretrained", choices=[False, True], default_value=True
        )
        multi_dropout = CategoricalHyperparameter(
            name="multi_dropout", choices=[True, False], default_value=False
        )
        activation_ff = CategoricalHyperparameter(
            name="activation_ff", choices=['gelu', 'silu'], default_value='silu'
        )
        num_layers = UniformIntegerHyperparameter(
            name="num_layers", lower=1, upper=4, default_value=2, log=False
        )

        cs.add_hyperparameters(
            [name, lr, max_epochs, pretrained, dropout, optimizer, lr_scheduler, weight_decay, multi_dropout,
             activation_ff, num_layers])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return (
                metadata.input_shape.width > 1 and metadata.input_shape.height > 1) and metadata.input_shape.max_sequence_len == 1 and (
            not UNET_Model.is_applicable(metadata))
