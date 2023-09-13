from ConfigSpace import Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    EqualsCondition
from skorch.callbacks import LRScheduler
from torch import nn
from torch.optim import AdamW, Adam

from .skorch_base import SkorchBaseModel


class MLP(nn.Module):

    def __init__(self, metadata, num_classes, activation_ff, dropout, n_layers, d_model):
        super(MLP, self).__init__()
        input_dim = metadata.input_shape.max_sequence_len * metadata.input_shape.channels * metadata.input_shape.height * metadata.input_shape.width
        layers = []
        if activation_ff == 'gelu':
            activation_class = nn.GELU
        elif activation_ff == 'relu':
            activation_class = nn.ReLU
        else:
            raise ValueError("Unknown activation function.")

        layer_dim = [input_dim] + [d_model] * (n_layers - 1) + [num_classes]
        print("initializing mlp with architecture", layer_dim)
        for i in range(n_layers):
            if i > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            if i < n_layers - 1:
                layers.append(activation_class())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.net(x)


class MLP_Model(SkorchBaseModel):
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
            MLP,
            max_epochs=self.cfg['max_epochs'],
            lr=self.cfg['lr'],
            # problem dimensions
            module__metadata=self.metadata,
            module__num_classes=self.output_shape[0],
            # model parameters
            module__d_model=self.cfg['d_model'],
            module__n_layers=self.cfg['n_layers'],
            module__dropout=self.cfg['dropout'],
            module__activation_ff=self.cfg['activation_ff'],
            callbacks=self.callbacks,
            **self.skorch_kwargs,
        )

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="mlp")
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
            name="weight_decay", lower=1e-7, upper=1e-1, default_value=0.03, log=True
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
            name="d_model", lower=32, upper=1024, default_value=64, log=False
        )
        activation_ff = CategoricalHyperparameter(
            name="activation_ff", choices=['gelu', 'relu'], default_value='gelu'
        )
        cs.add_hyperparameters(
            [name, lr, max_epochs, d_model, dropout, n_layers, weight_decay, lr_scheduler, optimizer, activation_ff])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return metadata.input_shape.width * metadata.input_shape.height * metadata.input_shape.max_sequence_len * metadata.input_shape.channels < 1024
