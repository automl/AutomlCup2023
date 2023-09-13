from ConfigSpace import Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    EqualsCondition
from skorch.callbacks import LRScheduler
from torch.optim import AdamW, Adam

from .neural_nets.multires import MultiresNet
from .skorch_base import SkorchBaseModel


class MultiResConvModel(SkorchBaseModel):
    def __init__(self, metadata, input_cfg=None, seed=None):
        super().__init__(metadata, input_cfg, seed)

        self.dtype_input = 'int' if self.metadata.input_shape.channels == 1 else 'float32'

        if self.cfg['optimizer'] == 'Adam':
            self.skorch_kwargs['optimizer'] = Adam
        elif self.cfg['optimizer'] == 'AdamW':
            self.skorch_kwargs['optimizer'] = AdamW
            self.skorch_kwargs['optimizer__weight_decay'] = self.cfg['weight_decay']

        if self.cfg['lr_scheduler']:
            self.callbacks.append(
                self.callbacks.append(LRScheduler(policy='CosineAnnealingLR', T_max=self.cfg['max_epochs'])))

        n_tokens = self.metadata.n_tokens if hasattr(self.metadata, 'n_tokens') else self.metadata.input_shape.channels
        self.model = self.model_class(
            MultiresNet,
            max_epochs=self.cfg['max_epochs'],
            lr=self.cfg['lr'],
            # problem dimensions
            module__d_input=self.metadata.input_shape.channels,
            module__d_output=self.output_shape[0],
            module__max_length=self.input_shape[0],
            module__n_tokens=n_tokens,
            # model parameters
            module__d_model=self.cfg['d_model'],
            module__n_layers=self.cfg['n_layers'],
            module__dropout=self.cfg['dropout'],
            module__kernel_size=self.cfg['kernel_size'],
            module__batchnorm=True,
            module__encoder='embedding' if self.metadata.input_shape.channels == 1 else 'linear',
            module__tree_select='fading',
            module__indep_res_init=True,
            callbacks=self.callbacks,
            **self.skorch_kwargs,
        )

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="multires")
        lr = UniformFloatHyperparameter(
            name="lr", lower=1e-5, upper=1e-2, default_value=3e-3, log=True
        )
        optimizer = CategoricalHyperparameter(
            name="optimizer", choices=['Adam', 'AdamW'], default_value='AdamW'
        )
        lr_scheduler = CategoricalHyperparameter(
            name="lr_scheduler", choices=[True, False], default_value=True
        )
        weight_decay = UniformFloatHyperparameter(
            name="weight_decay", lower=1e-5, upper=1e-1, default_value=0.03, log=True
        )
        dropout = UniformFloatHyperparameter(
            name="dropout", lower=0., upper=0.5, default_value=0.1, log=False
        )
        max_epochs = UniformIntegerHyperparameter(
            name="max_epochs", lower=1, upper=300, default_value=200, log=False
        )
        n_layers = UniformIntegerHyperparameter(  # 6 for submission 10 for paper
            name="n_layers", lower=1, upper=10, default_value=4, log=False
        )
        d_model = UniformIntegerHyperparameter(  # 104 for submission 128 for paper
            name="d_model", lower=32, upper=512, default_value=64, log=False
        )
        kernel_size = UniformIntegerHyperparameter(
            name="kernel_size", lower=2, upper=8, default_value=4, log=False
        )
        cs.add_hyperparameters(
            [name, lr, max_epochs, kernel_size, d_model, dropout, n_layers, weight_decay, lr_scheduler, optimizer])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return metadata.input_dimension == 1 and metadata.input_shape.max_sequence_len > 1
