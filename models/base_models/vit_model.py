"""
Reference : https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html
"""
import math

import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F
from ConfigSpace import EqualsCondition, Constant, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from skorch.callbacks import LRScheduler
from torch.optim import Adam, AdamW
from torchvision.models import ViT_B_32_Weights

from .skorch_base import SkorchBaseModel
from .unet import UNET_Model


class VIT(nn.Module):

    def __init__(self, metadata, num_classes, use_pretrained=True, dropout=0., attention_dropout=0., num_layers=2,
                 multi_dropout=False):
        super(VIT, self).__init__()
        self.metadata = metadata
        timeseries = self.metadata.input_shape.max_sequence_len
        channel = self.metadata.input_shape.channels
        self.net = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT if use_pretrained else None, dropout=dropout,
                                   attention_dropout=attention_dropout)

        num_ftrs = self.net.heads.head.in_features
        last_layers = []
        layer_dims = np.rint(np.logspace(np.log10(num_classes), np.log10(num_ftrs), num_layers + 1))[::-1].astype(int)
        for i in range(num_layers):
            last_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < num_layers - 1:  # not last layer of last layers
                last_layers.append(nn.GELU())
                if multi_dropout:
                    last_layers.append(nn.Dropout(dropout))
        self.net.heads.head = nn.Sequential(*last_layers)

        if not (timeseries * channel == 1 or timeseries * channel == 3):
            self.net.conv_proj = nn.Conv2d(timeseries * channel, 768, kernel_size=(32, 32), stride=(32, 32))

    def forward(self, x):
        # TODO fix for timeseries > 1 and channel=1 and make sure its correct in all cases 
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        x = x.reshape((x.shape[0], -1, x.shape[3], x.shape[4]))
        x = F.resize(x, 224)
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


class VIT_Model(SkorchBaseModel):
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
            VIT,
            module__metadata=metadata,
            module__num_classes=math.prod(self.output_shape),
            module__dropout=self.cfg['dropout'],
            module__attention_dropout=self.cfg['attention_dropout'],
            module__use_pretrained=self.cfg['pretrained'],
            module__multi_dropout=self.cfg['multi_dropout'],
            module__num_layers=self.cfg['num_layers'],
            max_epochs=self.cfg['max_epochs'],
            lr=self.cfg['lr'],
            callbacks=self.callbacks,
            **self.skorch_kwargs
        )

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="vit")
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
        attention_dropout = UniformFloatHyperparameter(
            name="attention_dropout", lower=0., upper=0.5, default_value=0.2, log=False
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
        num_layers = UniformIntegerHyperparameter(
            name="num_layers", lower=1, upper=4, default_value=2, log=False
        )
        cs.add_hyperparameters(
            [name, lr, max_epochs, pretrained, dropout, optimizer, lr_scheduler, weight_decay, attention_dropout,
             multi_dropout, num_layers])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return (
                metadata.input_shape.width >= 224 and metadata.input_shape.height >= 224) and metadata.input_shape.max_sequence_len == 1 and (
            not UNET_Model.is_applicable(metadata))
