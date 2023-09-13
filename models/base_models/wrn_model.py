import math

from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant, CategoricalHyperparameter, \
    EqualsCondition
from skorch.callbacks import LRScheduler
from torch.optim import Adam, AdamW

from .multires_conv_model import MultiResConvModel
from .skorch_base import SkorchBaseModel
from .unet import UNET_Model
from .wrn1d import WideResNet1d
from .wrn2d import WideResNet2d
from .wrn3d import WideResNet3d


class WRN_Model(SkorchBaseModel):
    def __init__(self, metadata, input_cfg=None, seed=None):
        super().__init__(metadata, input_cfg, seed)
        spacetime_dims = metadata.input_dimension  # np.count_nonzero(np.array(self.input_shape)[[0, 2, 3]] != 1)

        if spacetime_dims <= 1:
            wrn_class = WideResNet1d
        elif spacetime_dims == 2:
            wrn_class = WideResNet2d
        elif spacetime_dims == 3:
            wrn_class = WideResNet3d
        else:
            raise ValueError(f'Uknown input dimension. Given input dimension is {spacetime_dims}')
        # stratify = False

        if self.cfg['optimizer'] == 'Adam':
            self.skorch_kwargs['optimizer'] = Adam
        elif self.cfg['optimizer'] == 'AdamW':
            self.skorch_kwargs['optimizer'] = AdamW
            self.skorch_kwargs['optimizer__weight_decay'] = self.cfg['weight_decay']

        if self.cfg['lr_scheduler']:
            self.callbacks.append(
                self.callbacks.append(LRScheduler(policy='CosineAnnealingLR', T_max=self.cfg['max_epochs'])))

        self.model = self.model_class(
            wrn_class,
            module__input_shape=self.input_shape,
            module__num_classes=math.prod(self.output_shape),
            module__widen_factor=self.cfg["widen_factor"],
            module__dropRate=self.cfg["dropRate"],
            module__in_channels=self.input_shape[1] if spacetime_dims > 0 else 1,
            module__depth=self.cfg["depth"] * 6 + 4,
            max_epochs=self.cfg['max_epochs'],
            lr=self.cfg['lr'],
            callbacks=self.callbacks,
            **self.skorch_kwargs
        )

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="wrn")
        depth = UniformIntegerHyperparameter(
            name="depth", lower=1, upper=4, default_value=1, log=False
        )
        widen_factor = UniformIntegerHyperparameter(
            name="widen_factor", lower=2, upper=6, default_value=4, log=False
        )
        dropRate = UniformFloatHyperparameter(
            name="dropRate", lower=0.0, upper=0.3, default_value=0.0, log=False
        )
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
        cs.add_hyperparameters(
            [name, depth, widen_factor, dropRate, lr, max_epochs, lr_scheduler, weight_decay, optimizer])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return (metadata.input_dimension > 0) and (not UNET_Model.is_applicable(metadata)) and (
            not MultiResConvModel.is_applicable(metadata))
