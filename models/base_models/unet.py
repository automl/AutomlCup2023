"""
Reference : https://github.com/milesial/Pytorch-UNet
"""

import numpy as np
import torch
import torch.nn as nn
from ConfigSpace import EqualsCondition, UniformFloatHyperparameter, Constant, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from skorch.callbacks import LRScheduler
from torch.optim import Adam, AdamW

from .skorch_base import SkorchBaseModel


class PartialLabelLossWrapper(nn.Module):
    def __init__(self, original_loss):
        super().__init__()
        self.original_loss = original_loss

    def forward(self, yhat, y):
        nan_mask = torch.isnan(y)
        y[nan_mask] = yhat[nan_mask]
        loss = self.original_loss(yhat, y)
        return loss


class PretrainedUNet(nn.Module):

    def __init__(self, metadata, use_pretrained=True):
        super(PretrainedUNet, self).__init__()
        self.metadata = metadata
        timeseries = self.metadata.input_shape.max_sequence_len
        channel = self.metadata.input_shape.channels
        self.unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=use_pretrained,
                                   scale=1)  # scale??
        # self.disable_gradients(self.unet)
        if len(self.metadata.output_shape) == 2:
            output_channels = 1
        else:
            output_channels = self.metadata.output_shape[0]
        if output_channels != 3:  # TODO: test for output_channels=3
            self.unet.outc = nn.Conv2d(64, output_channels, kernel_size=(1, 1), stride=(1, 1))
        if not (timeseries * channel == 1 or timeseries * channel == 3):
            self.unet.inc.double_conv[0] = nn.Conv2d(timeseries * channel, 64, kernel_size=(3, 3), stride=(1, 1),
                                                     padding=(1, 1), bias=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.metadata.input_shape.width,
                      self.metadata.input_shape.height)  # TODO make sure width comes first
        # TODO fix for timeseries > 1 and channel=1 and make sure its correct in all cases 
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # x = torch.squeeze(x)
        x = self.unet.forward(x)
        x = x.reshape((x.shape[0], *self.metadata.output_shape))
        return x

    def enable_gradients(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = True

    def disable_gradients(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = False


class UNET_Model(SkorchBaseModel):
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

        self.skorch_kwargs["criterion"] = PartialLabelLossWrapper(self.skorch_kwargs["criterion"])
        self.model = self.model_class(
            PretrainedUNet,
            module__metadata=metadata,
            module__use_pretrained=self.cfg['pretrained'],
            max_epochs=self.cfg['max_epochs'],
            lr=self.cfg['lr'],
            callbacks=self.callbacks,
            **self.skorch_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_unlabeled: np.ndarray = None, y_unlabeled: np.ndarray = None,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        if (y_unlabeled is not None) and not (np.isnan(y_unlabeled)).all() and self.cfg['use_ssl']:
            X = np.concatenate((X, X_unlabeled))
            y = np.concatenate((y, y_unlabeled))
        super().fit(X, y, X_unlabeled, y_unlabeled, X_val, y_val)

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="unet")
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
        max_epochs = UniformIntegerHyperparameter(
            name="max_epochs", lower=1, upper=300, default_value=300, log=False
        )
        pretrained = CategoricalHyperparameter(
            name="pretrained", choices=[False, True], default_value=True
        )
        cs.add_hyperparameters([name, lr, max_epochs, pretrained, optimizer, lr_scheduler, weight_decay])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        if len(metadata.output_shape) > 1 and metadata.output_shape[-2] == metadata.input_shape.width and \
                metadata.output_shape[-1] == metadata.input_shape.height:
            return True
        else:
            return False
